"""Queue, staging, worker, and reporting checks without a real Slurm submit."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import gmsh
import numpy as np
from PIL import Image

from benchmark.run import (aggregate, array_spec, check_previous_jobs_finished, main, pending_indices,
                           parse_partitions, resolved_config, submit_array)
from benchmark.staging import selected_images, stage_images
from benchmark.worker import claim_next, convert_one, process_queue, write_json


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def image_file(self, path: Path, *, pore=True) -> Path:
        pixels = np.full((32, 32), 65535, dtype=np.uint16)
        if pore:
            pixels[10:16, 10:16] = 0
        Image.fromarray(pixels).save(path)
        return path

    def test_selection_and_explicit_area(self):
        records = selected_images()
        self.assertEqual(len(records), 300)
        self.assertEqual(len({r["id"] for r in records}), 300)
        self.assertEqual([sum(r["sample"] == sample for r in records)
                          for sample in ("HPC22", "HPC30", "HPC45")], [100] * 3)
        self.assertEqual(records[0]["id"], "HPC22_0000")
        self.assertEqual(records[-1]["id"], "HPC45_0198")
        self.assertEqual(array_spec(list(range(300))), "0-3%4")
        self.assertEqual(array_spec([1, 5, 8]), "0-2%4")
        with self.assertRaises(SystemExit):
            main(["--dry-run"])
        with self.assertRaises(ValueError):
            resolved_config((3300, 2200, 384, 384))
        self.assertIsNone(resolved_config(None)["segmentation"]["crop"])
        cache, results = self.root / "dry_cache", self.root / "dry_results"
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(main(["--crop", "1400", "900", "384", "384", "--dry-run",
                                   "--cache", str(cache), "--output-root", str(results)]), 0)
        plan = json.loads(stdout.getvalue())
        self.assertEqual(plan["task_count"], 300)
        self.assertEqual(plan["maximum_concurrent_nodes"], 4)
        self.assertEqual(plan["worker_count"], 4)
        self.assertEqual(plan["partition"], "share,dgx2")
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(main(["--crop", "1400", "900", "384", "384", "--dry-run",
                                   "--wait-for-all-nodes", "--partitions", "share,dgxh"]), 0)
        plan = json.loads(stdout.getvalue())
        self.assertTrue(plan["wait_for_all_nodes"])
        self.assertEqual(plan["partitions"], "share,dgxh")
        with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(main(["--crop", "1400", "900", "384", "384", "--dry-run",
                                   "--threshold-percentile", "30"]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["config"]["segmentation"]["threshold_percentile"], 30)
        self.assertFalse(cache.exists())
        self.assertFalse(results.exists())

    def test_partition_names_and_shell_quoting(self):
        self.assertEqual(parse_partitions("dgxh,dgx2,eecs,gpu,ampere"),
                         "dgxh,dgx2,eecs,gpu,ampere")
        with self.assertRaisesRegex(ValueError, "partitions"):
            parse_partitions('"dgxh","dgx2"')
        with self.assertRaisesRegex(ValueError, "partitions"):
            parse_partitions("share,share")

    def test_direct_script_launch_from_checkout(self):
        checkout = Path(__file__).resolve().parents[1]
        interpreters = [sys.executable]
        shell_python = shutil.which("python")
        if shell_python and shell_python not in interpreters:
            interpreters.append(shell_python)
        for interpreter in interpreters:
            with self.subTest(interpreter=interpreter):
                process = subprocess.run(
                    [interpreter, "benchmark/run.py", "--crop", "1400", "900", "384", "384", "--dry-run"],
                    cwd=checkout, capture_output=True, text=True,
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(json.loads(process.stdout)["task_count"], 300)

    def test_config_override_and_validation(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps({"segmentation": {"threshold": 0.4},
                                           "meshing": {"interior_size_nm": 90}}))
        config = resolved_config((0, 0, 32, 32), config_path)
        self.assertEqual(config["segmentation"]["threshold"], 0.4)
        self.assertEqual(config["meshing"]["interior_size_nm"], 90)
        self.assertEqual(resolved_config((0, 0, 32, 32), threshold_percentile=30)
                         ["segmentation"]["threshold_percentile"], 30)
        with self.assertRaisesRegex(ValueError, "threshold_percentile"):
            resolved_config((0, 0, 32, 32), threshold_percentile=100)
        config_path.write_text(json.dumps({"segmentation": {"crop": [0, 0, 32, 32]}}))
        with self.assertRaisesRegex(ValueError, "choose crop"):
            resolved_config(None, config_path)

    def test_stage_cache_crc_and_repair(self):
        record = selected_images()[0]
        image = self.image_file(self.root / "source.tif")
        archive = self.root / "source.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.write(image, record["archive_member"])
        cache = self.root / "cache"
        staged = stage_images([record], cache, archive=archive)
        self.assertEqual(len(staged), 1)
        cached = Path(staged[0]["path"])
        self.assertEqual(cached.read_bytes(), image.read_bytes())
        self.assertEqual(staged[0]["width"], 32)
        cached.write_bytes(b"corrupt")
        repaired = stage_images([record], cache, archive=archive)
        self.assertEqual(Path(repaired[0]["path"]).read_bytes(), image.read_bytes())
        self.assertEqual(repaired[0]["sha256"], staged[0]["sha256"])

    def make_run(self, *, missing_source=False):
        run_dir = self.root / "run"
        run_dir.mkdir()
        source = self.image_file(self.root / "image.tif")
        import hashlib
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        config = resolved_config((0, 0, 32, 32))
        config["segmentation"].update(min_solid_pixels=0, fill_holes_pixels=0)
        task = {"id": "HPC22_0000", "path": str(self.root / "missing.tif" if missing_source else source),
                "sha256": sha, "index": 0}
        write_json(run_dir / "run.json", {"config": config, "tasks": [task], "submissions": []})
        return run_dir

    def test_worker_mesh_png_report_resume_and_aggregate(self):
        run_dir = self.make_run()
        self.assertEqual(pending_indices(run_dir), [0])
        report = convert_one(run_dir, 0)
        self.assertEqual(report["status"], "complete", report.get("error"))
        folder = run_dir / "tasks" / "HPC22_0000"
        self.assertEqual(pending_indices(run_dir), [])
        with Image.open(folder / "mesh.png") as image:
            self.assertEqual(image.size, (2048, 2048))
            self.assertEqual(image.convert("L").getextrema(), (0, 255))
            self.assertNotEqual(image.getpixel((1024, 1024)), (0, 0, 0))
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(folder / "mesh.msh"))
            _, triangles = gmsh.model.mesh.getElementsByType(2)
            self.assertGreater(len(triangles), 0)
        finally:
            gmsh.finalize()
        summary = aggregate(run_dir)
        self.assertEqual(summary["completed"], 1)
        self.assertGreater(summary["conversion_baseline"]["wall_seconds"], 0)
        self.assertGreater(summary["conversion_baseline"]["images_per_second"], 0)
        with (run_dir / "measurements.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row["status"], "complete")
        self.assertGreater(float(row["mesh_seconds"]), 0)
        (folder / "mesh.png").unlink()
        self.assertEqual(pending_indices(run_dir), [0])

    def test_failed_worker_and_missing_report(self):
        run_dir = self.make_run(missing_source=True)
        report = convert_one(run_dir, 0)
        self.assertEqual(report["status"], "failed")
        self.assertIn("FileNotFoundError", report["error"])
        self.assertEqual(pending_indices(run_dir), [0])
        self.assertEqual(aggregate(run_dir)["failed"], 1)

    def test_percentile_worker_records_actual_cutoff(self):
        run_dir = self.make_run()
        path = run_dir / "run.json"
        manifest = json.loads(path.read_text())
        manifest["config"]["segmentation"]["threshold_percentile"] = 30
        write_json(path, manifest)
        report = convert_one(run_dir, 0)
        self.assertEqual(report["status"], "complete", report.get("error"))
        self.assertEqual(report["metadata"]["threshold_mode"], "percentile")
        self.assertEqual(report["metadata"]["threshold_percentile"], 30)
        self.assertGreater(report["metadata"]["threshold_raw"], 0)

    def test_workers_claim_each_image_once_and_continue_after_failure(self):
        run_dir = self.make_run()
        queue_path = run_dir / "queue.json"
        indices = list(range(31))
        write_json(queue_path, {"indices": indices, "next": 0})
        seen = []

        def fake_convert(_run_dir, index):
            seen.append(index)
            return {"id": str(index), "status": "failed" if index == 7 else "complete"}

        with mock.patch("benchmark.worker.convert_one", side_effect=fake_convert):
            with mock.patch("sys.stdout", new_callable=io.StringIO):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    totals = list(pool.map(lambda _: process_queue(run_dir, queue_path), range(4)))
        self.assertEqual(sorted(seen), indices)
        self.assertEqual(sum(total["completed"] for total in totals), 30)
        self.assertEqual(sum(total["failed"] for total in totals), 1)
        self.assertIsNone(claim_next(queue_path))

    def test_persistent_worker_meshes_multiple_images_in_one_process(self):
        run_dir = self.make_run()
        manifest_path = run_dir / "run.json"
        manifest = json.loads(manifest_path.read_text())
        second = {**manifest["tasks"][0], "id": "HPC22_0002", "index": 1}
        manifest["tasks"].append(second)
        write_json(manifest_path, manifest)
        queue_path = run_dir / "queue.json"
        write_json(queue_path, {"indices": [0, 1], "next": 0})
        with mock.patch("sys.stdout", new_callable=io.StringIO):
            result = process_queue(run_dir, queue_path)
        self.assertEqual(result, {"completed": 2, "failed": 0})
        self.assertEqual(pending_indices(run_dir), [])

    def test_submit_uses_exclusive_array_and_afterany(self):
        run_dir = self.make_run()
        outputs = iter(("12345\n", "12346\n"))
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

        with mock.patch("benchmark.run.subprocess.run", side_effect=fake_run):
            submission = submit_array(run_dir, [0])
        self.assertEqual(submission["array_job_id"], "12345")
        self.assertEqual(submission["aggregate_job_id"], "12346")
        self.assertIn("--exclusive", commands[0])
        self.assertIn("--partition=share,dgx2", commands[0])
        self.assertIn("--array=0-0%4", commands[0])
        self.assertIn("--nodes=1", commands[0])
        self.assertIn("--dependency=afterany:12345", commands[1])
        self.assertTrue(commands[0][-1].endswith("worker.sbatch"))
        self.assertTrue(commands[1][-1].endswith("aggregate.sbatch"))
        self.assertIn("-m benchmark.run work-queue", (run_dir / "worker.sbatch").read_text())
        queue_path = Path(submission["queue_path"])
        self.assertEqual(json.loads(queue_path.read_text()), {"indices": [0], "next": 0})
        self.assertIn(str(queue_path), (run_dir / "worker.sbatch").read_text())
        self.assertEqual(submission["worker_count"], 1)
        self.assertNotIn("--wrap", " ".join(commands[0]))
        self.assertEqual(len(json.loads((run_dir / "run.json").read_text())["submissions"]), 1)

    def test_submit_reports_slurm_stderr(self):
        run_dir = self.make_run()
        failure = subprocess.CompletedProcess([], 255, stdout="", stderr="sbatch: error: invalid request")
        with mock.patch("benchmark.run.subprocess.run", return_value=failure):
            with self.assertRaisesRegex(RuntimeError, "sbatch: error: invalid request"):
                submit_array(run_dir, [0])
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["submissions"], [])

    def test_wait_for_all_nodes_uses_one_allocation_and_srun(self):
        run_dir = self.make_run()
        outputs = iter(("12345\n", "12346\n"))
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=next(outputs), stderr="")

        with mock.patch("benchmark.run.subprocess.run", side_effect=fake_run):
            submission = submit_array(run_dir, [0, 1, 2, 3], wait_for_all_nodes=True,
                                      partitions="share,dgxh")
        self.assertEqual(submission["worker_count"], 4)
        self.assertTrue(submission["wait_for_all_nodes"])
        self.assertIn("--partition=share,dgxh", commands[0])
        self.assertIn("--nodes=4", commands[0])
        self.assertIn("--ntasks=4", commands[0])
        self.assertIn("--wait-all-nodes=1", commands[0])
        self.assertFalse(any(arg.startswith("--array=") for arg in commands[0]))
        script = (run_dir / "worker.sbatch").read_text()
        self.assertIn("exec srun --nodes=4 --ntasks=4 --ntasks-per-node=1", script)

    def test_resume_after_submission_failure_does_not_stage_again(self):
        run_dir = self.make_run()
        responses = iter(("", "12345\n", "12346\n"))

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout=next(responses), stderr="")

        with mock.patch("benchmark.run.stage_images", side_effect=AssertionError("cache touched")):
            with mock.patch("benchmark.run.subprocess.run", side_effect=fake_run):
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    self.assertEqual(main(["--resume", str(run_dir)]), 0)
        submissions = json.loads((run_dir / "run.json").read_text())["submissions"]
        self.assertEqual(submissions[0]["array_job_id"], "12345")

    def test_resume_rejects_active_array_or_aggregation(self):
        run_dir = self.make_run()
        manifest_path = run_dir / "run.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["submissions"] = [{"array_job_id": "12345", "aggregate_job_id": "12346"}]
        write_json(manifest_path, manifest)
        for active in ("12345_[0-1%4]\n", "12346\n"):
            with self.subTest(active=active):
                output = subprocess.CompletedProcess([], 0, stdout=active, stderr="")
                with mock.patch("benchmark.run.subprocess.run", return_value=output):
                    with self.assertRaisesRegex(RuntimeError, "still active"):
                        check_previous_jobs_finished(run_dir)
        output = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with mock.patch("benchmark.run.subprocess.run", return_value=output):
            check_previous_jobs_finished(run_dir)


if __name__ == "__main__":
    unittest.main()
