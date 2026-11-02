import contextlib
import csv
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import random
import statistics
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from pore_generator.__main__ import main
from pore_generator.generation import Config, generate, sample_radius_mm


class PoreGeneratorTests(unittest.TestCase):
    def test_reproducibility(self):
        config = Config()
        first = generate(config, 42)
        self.assertEqual(first, generate(config, 42))
        self.assertNotEqual(first["pores"], generate(config, 43)["pores"])

    def test_geometry_and_porosity(self):
        config = Config(clearance_um=1)
        sample = generate(config, 42)
        pores = sample["pores"]
        areas = {"explicit": 0.0, "small": 0.0}
        gap = config.clearance_um * 0.001
        for i, pore in enumerate(pores):
            r, x, y = pore["radius_mm"], pore["x_mm"], pore["y_mm"]
            self.assertGreaterEqual(r, config.min_radius_um * 0.001)
            self.assertGreater(min(x, y, config.size_mm - x, config.size_mm - y), r + gap)
            self.assertEqual(pore["kind"], "explicit" if r >= 0.020 else "small")
            for other in pores[:i]:
                self.assertGreater(math.hypot(x - other["x_mm"], y - other["y_mm"]),
                                   r + other["radius_mm"] + gap)
            areas[pore["kind"]] += math.pi * r ** 2
        total = sum(areas.values()) / config.size_mm ** 2
        self.assertGreaterEqual(total, config.target_porosity)
        self.assertLess(total - math.pi * pores[-1]["radius_mm"] ** 2 / config.size_mm ** 2,
                        config.target_porosity)
        self.assertAlmostEqual(total, sample["statistics"]["achieved_porosity"])
        for kind in areas:
            self.assertGreater(areas[kind], 0)
            self.assertAlmostEqual(areas[kind] / config.size_mm ** 2,
                                   sample["statistics"][kind + "_porosity"])

    def test_truncated_log_radius_distribution_and_units(self):
        # Raise the cutoff to exercise truncation; test draws before placement bias.
        config = Config(min_radius_um=20)
        rng = random.Random(10)
        logs = [math.log(sample_radius_mm(rng, config) * 1000) for _ in range(5000)]
        normal = statistics.NormalDist()
        alpha = (math.log(20) - config.radius_mu) / config.radius_sigma
        expected_mean = config.radius_mu + config.radius_sigma * normal.pdf(alpha) / (1 - normal.cdf(alpha))
        self.assertGreaterEqual(min(logs), math.log(20))
        self.assertAlmostEqual(statistics.mean(logs), expected_mean, delta=0.03)
        self.assertGreater(statistics.stdev(logs), 0.2)

    def test_invalid_configuration(self):
        for values in ({"size_mm": 0}, {"radius_mu": math.nan}, {"radius_sigma": -1},
                       {"clearance_um": -1}, {"target_porosity": 1}, {"mesh_size_um": 0},
                       {"min_radius_um": 21}, {"size_mm": 0.001}, {"size_mm": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Config(**values)

    def test_sampling_and_placement_failures(self):
        with patch("pore_generator.generation.random.Random.lognormvariate", return_value=600):
            with self.assertRaisesRegex(ValueError, "cannot fit"):
                generate(Config(), 42)
        with patch("pore_generator.generation.random.Random.lognormvariate", return_value=20), \
                patch("pore_generator.generation.random.Random.uniform", return_value=0.5):
            with self.assertRaisesRegex(ValueError, "could not place.*10000"):
                generate(Config(), 42)
        with patch("pore_generator.generation.random.Random.lognormvariate", return_value=0.5):
            with self.assertRaisesRegex(ValueError, "minimum after 10000"):
                generate(Config(), 42)
        self.assertEqual(generate(replace(Config(), target_porosity=0), 42)["pores"], [])

    def test_cli_files_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text('{"target_porosity": 0, "size_mm": 2}', encoding="utf-8")
            output = root / "sample"
            args = ["generate", "--seed", "42", "--output", str(output), "--config", str(config),
                    "--target-porosity", "0.05"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            sample = json.loads((output / "sample.json").read_text())
            self.assertEqual(sample["config"]["target_porosity"], 0.05)
            self.assertEqual(sample["config"]["size_mm"], 2)
            with (output / "pores.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), sample["statistics"]["pore_count"])
            self.assertEqual(float(rows[0]["radius_mm"]), sample["pores"][0]["radius_mm"])
            ET.parse(output / "preview.svg")
            original = (output / "sample.json").read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                main(args)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(original, (output / "sample.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
