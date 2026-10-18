from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from pore_generator.export import write_sample
from pore_generator.generation import Config, generate
from pore_generator.meshing import mesh_material


@unittest.skipUnless(importlib.util.find_spec("gmsh"), "Optional Gmsh is not installed")
class MaterialMeshCheck(unittest.TestCase):
    def test_mesh_export_and_inspection(self):
        # One small end-to-end check, including a square with no explicit pores.
        with tempfile.TemporaryDirectory() as directory:
            for porosity in (0.03, 0):
                with self.subTest(porosity=porosity):
                    output = Path(directory) / str(porosity)
                    config = replace(Config(), target_porosity=porosity, mesh_size_um=40)
                    write_sample(generate(config, 42), output)
                    summary = mesh_material(output)
                    self.assertEqual(summary["matrix_components"], 1)
                    self.assertGreater(summary["minimum_quality"], 0)
                    self.assertGreater(summary["minimum_triangle_area_mm2"], 0)
                    self.assertEqual(summary["interface_edges"] > 0, porosity > 0)
                    self.assertEqual(summary["pore_triangles"] > 0, porosity > 0)
                    self.assertAlmostEqual(summary["matrix_area_mm2"] + summary["pore_area_mm2"], 1)
                    mesh = output / "material_mesh"
                    with (mesh / "specimen.msh").open() as stream:
                        self.assertEqual(stream.readline(), "$MeshFormat\n")
                        self.assertEqual(stream.readline(), "4.1 0 8\n")
                    recorded = json.loads((mesh / "mesh_summary.json").read_text())
                    self.assertEqual(recorded["triangle_count"], summary["triangle_count"])
                    self.assertTrue((mesh / "mesh_preview.svg").is_file())
                    with self.assertRaisesRegex(ValueError, "already exists"):
                        mesh_material(output)


if __name__ == "__main__":
    unittest.main()
