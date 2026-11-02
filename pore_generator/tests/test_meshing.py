import contextlib
from dataclasses import replace
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from pore_generator.__main__ import main
from pore_generator.export import write_sample
from pore_generator.generation import Config, generate
from pore_generator.meshing import mesh_sample


@unittest.skipUnless(importlib.util.find_spec("gmsh"), "Optional Gmsh is not installed")
class MeshCheck(unittest.TestCase):
    def test_mesh_export_and_inspection(self):
        # One end-to-end check for both representations, including a solid square.
        with tempfile.TemporaryDirectory() as directory:
            for porosity in (0.03, 0):
                with self.subTest(porosity=porosity):
                    output = Path(directory) / str(porosity)
                    config = replace(Config(), target_porosity=porosity, mesh_size_um=40)
                    write_sample(generate(config, 42), output)
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(main(["mesh", "--sample", str(output), "--format", "both"]), 0)
                    summaries = {}
                    for mesh_format in ("material", "perforated"):
                        mesh = output / f"{mesh_format}_mesh"
                        summary = json.loads((mesh / "mesh_summary.json").read_text())
                        summaries[mesh_format] = summary
                        self.assertEqual(summary["matrix_components"], 1)
                        self.assertGreater(summary["minimum_quality"], 0)
                        self.assertGreater(summary["minimum_triangle_area_mm2"], 0)
                        self.assertEqual(summary["pore_boundary_edges"] > 0, porosity > 0)
                        self.assertEqual(summary["pore_triangles"] > 0, porosity > 0 and mesh_format == "material")
                        self.assertAlmostEqual(summary["matrix_area_mm2"] + summary["pore_area_mm2"]
                                               + summary["hole_area_mm2"], 1)
                        with (mesh / "specimen.msh").open() as stream:
                            self.assertEqual(stream.readline(), "$MeshFormat\n")
                            self.assertEqual(stream.readline(), "4.1 0 8\n")
                        self.assertTrue((mesh / "mesh_preview.svg").is_file())
                        with self.assertRaisesRegex(ValueError, "already exists"):
                            mesh_sample(output, mesh_format)
                    self.assertAlmostEqual(summaries["material"]["pore_area_mm2"],
                                           summaries["perforated"]["hole_area_mm2"])
                    mesh = output / "perforated_mesh"
                    metadata = json.loads((mesh / "geometry.json").read_text())
                    self.assertTrue(metadata["ready_for_fracture"])
                    self.assertFalse(metadata["small_pore_correction_applied"])
                    self.assertEqual((metadata["width"], metadata["height"]), (1, 1))
                    self.assertAlmostEqual(metadata["porosity"], summaries["perforated"]["hole_area_mm2"])
                    self.assertAlmostEqual(metadata["explicit_porosity"] + metadata["small_porosity"],
                                           metadata["total_sample_porosity"])
                    self.assertEqual(metadata["geometry_hash"], hashlib.sha256((mesh / "specimen.geo").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
