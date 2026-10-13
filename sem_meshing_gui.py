"""Local browser GUI for the Skarberg SEM dataset; no simulation runs in the GUI."""
from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from scientific_common import load_config, write_json
from sem_geometry import DEFAULTS, dataset_record, prepare_image, validate_params

ROOT = Path(__file__).resolve().parent


def processes_using_tcp_port(port):
    """Return process IDs using a TCP port, as reported by Linux fuser."""
    result = subprocess.run(
        ["fuser", "-n", "tcp", str(port)], capture_output=True, text=True
    )
    if result.returncode == 1:
        return []
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Could not inspect TCP port {port}: {detail}")
    return sorted({int(value) for value in result.stdout.split() if value.isdigit()})


def release_tcp_port(port, timeout=3.0):
    """Stop existing processes on *port* so this GUI can replace them."""
    pids = [pid for pid in processes_using_tcp_port(port) if pid != os.getpid()]
    if not pids:
        return
    print(f"Port {port} is in use by PID(s) {', '.join(map(str, pids))}; stopping them.", flush=True)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError(f"Cannot stop PID {pid} on port {port}: permission denied") from exc

    deadline = time.monotonic() + timeout
    remaining = pids
    while remaining and time.monotonic() < deadline:
        time.sleep(0.05)
        remaining = [pid for pid in processes_using_tcp_port(port) if pid != os.getpid()]
    if remaining:
        print(f"PID(s) {', '.join(map(str, remaining))} did not stop; sending SIGKILL.", flush=True)
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                raise RuntimeError(f"Cannot kill PID {pid} on port {port}: permission denied") from exc
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and processes_using_tcp_port(port):
            time.sleep(0.05)
    remaining = [pid for pid in processes_using_tcp_port(port) if pid != os.getpid()]
    if remaining:
        raise RuntimeError(f"TCP port {port} is still in use by PID(s) {remaining}")


def fracture_config(meta):
    c = load_config()
    c["geometry"].update(preset="solid", width=meta["width"], height=meta["height"])
    c["mesh"]["size"] = meta["parameters"]["bulk_size_nm"]*1e-6
    c["material"].update(ell=meta["parameters"]["ell_nm"]*1e-6, Gc=1e-5,
                          thickness=meta["width"])
    # The nm-scale damage residual is much smaller than the mm demo residual.
    # Retain relative convergence rather than accepting an absolute floor.
    c["solver"].update(ksp_atol=1e-24, snes_atol=1e-18)
    c["loading"].update(maximum_separation=0.02*meta["width"], increments=80,
                          minimum_increment=0.02*meta["width"]/80/128)
    c["output"].update(save_linear_systems=False)
    c["image_model_note"] = "Illustrative material/thickness/loading; 2D crop boundaries are assumed grips/free edges, not experimentally measured boundaries"
    return c


def png_url(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode()


class App:
    def __init__(self, dataset, output):
        self.dataset, self.output = Path(dataset).resolve(), Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.jobs, self.processes, self.lock = {}, {}, threading.Lock()
        for path in self.output.glob("*/job.json"):
            try:
                job = json.loads(path.read_text())
                if job.get("id") == path.parent.name and len(job["id"]) == 32 and all(c in "0123456789abcdef" for c in job["id"]) and job.get("status") in ("complete", "partial", "failed", "cancelled"):
                    self.jobs[job["id"]] = job
            except (ValueError, KeyError):
                continue

    def catalog(self):
        manifest = json.loads((self.dataset/"manifest.json").read_text())
        records = []
        for record in manifest["images"]:
            _, r = dataset_record(self.dataset, record["id"])
            with Image.open(self.dataset/r["image"]) as image:
                records.append({"id": r["id"], "sample": r["sample"], "slice_index": r["slice_index"],
                                "width": image.width, "height": image.height})
        return {"images": records, "defaults": DEFAULTS, "doi": manifest["doi"],
                "mask_kind": manifest["mask_kind"]}

    def preview(self, params):
        image, solid, meta = prepare_image(self.dataset, params)
        lo, hi = np.percentile(image, [1, 99])
        gray = np.uint8(np.clip((image-lo)/max(hi-lo, 1e-12), 0, 1)*255)
        counts, _ = np.histogram(image, bins=96, range=(0.0, 1.0))
        return {"image": png_url(gray), "mask": png_url(np.uint8(solid)*255),
                "intensity_histogram": counts.tolist(), "metadata": meta}

    def build(self, params):
        params = validate_params(params)
        dataset_record(self.dataset, params["image_id"])
        with self.lock:
            if any(j["status"] in ("running", "rendering") for j in self.jobs.values()):
                raise ValueError("A mesh or simulation operation is already running")
            key = uuid.uuid4().hex
            folder = self.output/key
            folder.mkdir()
            write_json(folder/"parameters.json", params)
            self.jobs[key] = {"id": key, "status": "running", "parameters": params}
        threading.Thread(target=self.worker, args=(key, folder), daemon=True).start()
        return self.jobs[key].copy()

    def fetch_slice(self, request):
        sample, index = request.get("sample"), request.get("index")
        if isinstance(sample, bool) or sample not in (22, 30, 45) or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 200:
            raise ValueError("Choose sample 22/30/45 and slice 0..199")
        with self.lock:
            if any(j["status"] in ("running", "rendering") for j in self.jobs.values()):
                raise ValueError("A mesh or dataset operation is already running")
            key = uuid.uuid4().hex
            folder = self.output/key
            folder.mkdir()
            self.jobs[key] = {"id": key, "kind": "download", "status": "running"}
        def worker():
            try:
                with (folder/"download.log").open("w") as log:
                    result = subprocess.run([sys.executable, str(ROOT/"fetch_skarberg.py"), "--output", str(self.dataset),
                               "--samples", str(sample), "--indices", str(index)], stdout=log, stderr=subprocess.STDOUT, timeout=300)
                if result.returncode:
                    raise RuntimeError((folder/"download.log").read_text()[-2000:])
                with self.lock:
                    self.jobs[key].update(status="complete", image_id=f"HPC{sample}_{index:04d}")
            except Exception as exc:
                with self.lock:
                    self.jobs[key].update(status="failed", error=str(exc))
            write_json(folder/"job.json", self.jobs[key])
        threading.Thread(target=worker, daemon=True).start()
        return self.jobs[key].copy()

    def simulate(self, request):
        mesh_id = request.get("mesh_id")
        backend = request.get("backend", "cpu")
        scale = request.get("deformation_scale", 5.0)
        fps = request.get("fps", 12)
        if backend not in ("cpu", "gpu"):
            raise ValueError("Simulation backend must be cpu or gpu")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or not 1 <= scale <= 50:
            raise ValueError("Deformation scale must be between 1 and 50")
        if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 60:
            raise ValueError("Animation FPS must be between 1 and 60")
        with self.lock:
            mesh_job = self.jobs.get(mesh_id, {}).copy()
            if not mesh_job or mesh_job.get("kind") in ("download", "simulation") or mesh_job.get("status") != "complete":
                raise ValueError("Select a completed mesh build")
            metadata = mesh_job.get("metadata", {})
            if not metadata.get("ready_for_fracture"):
                raise ValueError("The mesh needs one connected solid component spanning both grips")
            if any(j["status"] in ("running", "rendering") for j in self.jobs.values()):
                raise ValueError("A mesh or simulation operation is already running")
            key = uuid.uuid4().hex
            folder = self.output/key
            folder.mkdir()
            config = json.loads((self.output/mesh_id/"fracture_config.json").read_text())
            config["solver"]["backend"] = backend
            config["output"].update(save_fields=True, field_stride=1, save_linear_systems=False)
            write_json(folder/"simulation_config.json", config)
            self.jobs[key] = {"id": key, "kind": "simulation", "status": "running",
                              "mesh_id": mesh_id, "backend": backend,
                              "deformation_scale": float(scale), "fps": fps,
                              "progress": 0.0, "accepted_steps": 0,
                              "maximum_separation": config["loading"]["maximum_separation"],
                              "geometry_hash": metadata["geometry_hash"]}
            write_json(folder/"job.json", self.jobs[key])
        threading.Thread(target=self.simulation_worker, args=(key, folder), daemon=True).start()
        return self.jobs[key].copy()

    def simulation_worker(self, key, folder):
        job = self.jobs[key]
        run = folder/"run"
        mesh = self.output/job["mesh_id"]/"mesh"
        command = [sys.executable, str(ROOT/"simulate_fracture.py"), "--config", str(folder/"simulation_config.json"),
                   "--output", str(run), "--image-mesh", str(mesh), "--backend", job["backend"]]
        environment = os.environ.copy()
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")
        if job["backend"] == "gpu" and "use_gpu_aware_mpi" not in environment.get("PETSC_OPTIONS", ""):
            environment["PETSC_OPTIONS"] = (environment.get("PETSC_OPTIONS", "") + " -use_gpu_aware_mpi 0").strip()
        try:
            with (folder/"simulation.log").open("w") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, bufsize=1, env=environment)
                with self.lock:
                    self.processes[key] = process
                for line in process.stdout:
                    log.write(line); log.flush()
                    match = re.search(r"step=(\d+) separation=([0-9.eE+\-]+).*damage=([0-9.eE+\-]+)", line)
                    if match:
                        step, separation, damage = int(match.group(1)), float(match.group(2)), float(match.group(3))
                        with self.lock:
                            self.jobs[key].update(accepted_steps=step, separation=separation,
                                                  damage_max=damage, progress=min(0.94, 0.94*separation/job["maximum_separation"]))
                            write_json(folder/"job.json", self.jobs[key])
                returncode = process.wait()
                with self.lock:
                    self.processes.pop(key, None)
            reason = None
            if returncode:
                reason = "Fracture solver failed"
                status = run/"status.json"
                if status.exists():
                    reason = json.loads(status.read_text()).get("reason", reason)
                frames_path, history_path = run/"frames.json", run/"history.csv"
                if not frames_path.exists() or not history_path.exists() or not any(
                        frame["step"] > 0 for frame in json.loads(frames_path.read_text())):
                    raise RuntimeError(reason)
            self.render_simulation(key, folder, reason)
        except Exception as exc:
            with self.lock:
                process = self.processes.pop(key, None)
            if process and process.poll() is None:
                process.kill()
            with self.lock:
                self.jobs[key].update(status="failed", error=str(exc))
        write_json(folder/"job.json", self.jobs[key])

    def render_simulation(self, key, folder, stopping_reason=None):
        """Render saved accepted fields, preserving unsuccessful solve status."""
        job = self.jobs[key]
        run = folder/"run"
        try:
            with self.lock:
                self.jobs[key].update(status="rendering", progress=0.96, stopping_reason=stopping_reason)
                write_json(folder/"job.json", self.jobs[key])
            visual = subprocess.Popen([sys.executable, str(ROOT/"visualize.py"), str(run),
                                      "--output", str(run/"visualization"),
                                      "--deformation-scale", str(job["deformation_scale"]), "--fps", str(job["fps"]),
                                      *(["--allow-partial"] if stopping_reason else [])],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with self.lock:
                self.processes[key] = visual
            visual_output, _ = visual.communicate(timeout=1800)
            with self.lock:
                self.processes.pop(key, None)
            (folder/"visualization.log").write_text(visual_output)
            if visual.returncode:
                raise RuntimeError("Animation rendering failed: " + visual_output[-1000:])
            if stopping_reason:
                with (run/"history.csv").open() as stream:
                    records = list(csv.DictReader(stream))
                metadata = json.loads((run/"metadata.json").read_text())
                summary = {"status": "partial", "dofs": metadata["dofs"],
                           "accepted_steps": len(records), "final_damage_max": float(records[-1]["damage_max"]),
                           "geometry_hash": metadata["geometry_hash"], "stopping_reason": stopping_reason}
                write_json(run/"partial_summary.json", summary)
            else:
                summary = json.loads((run/"summary.json").read_text())
            frames = json.loads((run/"frames.json").read_text())
            with self.lock:
                self.jobs[key].update(status="partial" if stopping_reason else "complete", progress=1.0, summary=summary,
                                      frame_count=len(frames), accepted_steps=summary["accepted_steps"],
                                      damage_max=summary["final_damage_max"])
        except Exception as exc:
            with self.lock:
                process = self.processes.pop(key, None)
            if process and process.poll() is None:
                process.kill()
            with self.lock:
                self.jobs[key].update(status="failed", error=str(exc))
        write_json(folder/"job.json", self.jobs[key])

    def shutdown(self):
        with self.lock:
            processes = list(self.processes.values())
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def worker(self, key, folder):
        try:
            with (folder/"build.log").open("w") as log:
                result = subprocess.run([sys.executable, str(ROOT/"sem_geometry.py"),
                                        "--dataset", str(self.dataset), "--parameters", str(folder/"parameters.json"),
                                        "--output", str(folder/"mesh")], stdout=log, stderr=subprocess.STDOUT, timeout=300)
            if result.returncode:
                raise RuntimeError((folder/"build.log").read_text()[-2500:])
            meta = json.loads((folder/"mesh/geometry.json").read_text())
            write_json(folder/"fracture_config.json", fracture_config(meta))
            with self.lock:
                self.jobs[key].update(status="complete", metadata=meta)
        except Exception as exc:
            with self.lock:
                self.jobs[key].update(status="failed", error=str(exc))
        write_json(folder/"job.json", self.jobs[key])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, body, mime="application/json"):
        data = json.dumps(body, allow_nan=False).encode() if mime == "application/json" else body
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            if urlparse("http://"+self.headers.get("Host", "")).hostname not in ("127.0.0.1", "localhost"):
                return self.send(403, {"error": "Localhost requests only"})
            path = urlparse(self.path)
            query = parse_qs(path.query)
            app = self.server.app
            if path.path == "/":
                return self.send(200, (ROOT/"gui/sem_meshing.html").read_bytes(), "text/html; charset=utf-8")
            if path.path == "/sem_meshing.js":
                return self.send(200, (ROOT/"gui/sem_meshing.js").read_bytes(), "text/javascript; charset=utf-8")
            if path.path == "/api/catalog":
                return self.send(200, app.catalog())
            if path.path == "/api/simulation/video":
                key = query.get("id", [""])[0]
                with app.lock:
                    job = app.jobs.get(key, {}).copy()
                if job.get("kind") != "simulation" or job.get("status") not in ("complete", "partial"):
                    raise ValueError("Simulation animation not available")
                return self.send(200, (app.output/key/"run/visualization/fracture.mp4").read_bytes(), "video/mp4")
            if path.path in ("/api/job", "/api/mesh", "/api/download"):
                key = query.get("id", [""])[0]
                with app.lock:
                    job = app.jobs.get(key, {}).copy()
                if not job:
                    raise ValueError("Unknown mesh job")
                if path.path == "/api/job":
                    return self.send(200, job)
                if job.get("kind") == "simulation":
                    raise ValueError("Simulation artifacts use the simulation endpoints")
                if job["status"] != "complete":
                    raise ValueError("Mesh job is not complete")
                if job.get("kind") == "download":
                    raise ValueError("This job fetched a dataset slice; select a mesh build for artifacts")
                if path.path == "/api/mesh":
                    with np.load(app.output/key/"mesh/mesh.npz") as data:
                        points = np.ascontiguousarray(data["points"], dtype="<f4")
                        triangles = np.ascontiguousarray(data["triangles"], dtype="<u4")
                    header = struct.pack("<4sII", b"MSH1", len(points), len(triangles))
                    return self.send(200, header + points.tobytes() + triangles.tobytes(),
                                     "application/octet-stream")
                filename = query.get("file", [""])[0]
                allowed = {"mesh/specimen.msh", "mesh/geometry.json", "mesh/contours.json", "mesh/solid_mask.png", "fracture_config.json", "parameters.json"}
                if filename not in allowed:
                    raise ValueError("Unknown artifact")
                return self.send(200, (app.output/key/filename).read_bytes(), "application/octet-stream")
            self.send(404, {"error": "Not found"})
        except Exception as exc:
            self.send(400, {"error": str(exc)})

    def do_POST(self):
        try:
            if urlparse("http://"+self.headers.get("Host", "")).hostname not in ("127.0.0.1", "localhost"):
                return self.send(403, {"error": "Localhost requests only"})
            # Browser writes must originate from this local app.
            if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
                raise ValueError("Cross-origin requests are not accepted")
            n = int(self.headers.get("Content-Length", "0"))
            if not 0 < n <= 65536:
                raise ValueError("Invalid request length")
            params = json.loads(self.rfile.read(n))
            if self.path == "/api/preview":
                self.send(200, self.server.app.preview(params))
            elif self.path == "/api/build":
                self.send(202, self.server.app.build(params))
            elif self.path == "/api/fetch":
                self.send(202, self.server.app.fetch_slice(params))
            elif self.path == "/api/simulate":
                self.send(202, self.server.app.simulate(params))
            else:
                self.send(404, {"error": "Not found"})
        except Exception as exc:
            self.send(400, {"error": str(exc)})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("datasets/skarberg"))
    p.add_argument("--output", type=Path, default=Path("results/sem_gui"))
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    if not 1 <= a.port <= 65535:
        p.error("--port must be between 1 and 65535")
    app = App(a.dataset, a.output)
    app.catalog()
    release_tcp_port(a.port)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    server.app = app
    def stop_server(signum, frame):
        app.shutdown()
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_server)
    print(f"SEM meshing GUI: http://127.0.0.1:{a.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        app.shutdown()
        server.server_close()
