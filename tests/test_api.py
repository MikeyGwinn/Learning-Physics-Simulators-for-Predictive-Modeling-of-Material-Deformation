"""Behavioral checks for the public SEM image-to-mesh pipeline."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from sem2msh import MeshingOptions, SEMImage, SegmentationOptions


class SEMImageTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def write_image(self, pixels, name="input.png"):
        path = self.root / name
        Image.fromarray(pixels).save(path)
        return path

    def test_8_and_16_bit_threshold_and_crop(self):
        for dtype, maximum in ((np.uint8, 255), (np.uint16, 65535)):
            with self.subTest(dtype=dtype):
                pixels = np.full((8, 10), maximum, dtype=dtype)
                pixels[2:5, 3:6] = 0
                suffix = "tif" if dtype is np.uint16 else "png"
                path = self.write_image(pixels, f"source_{maximum}.{suffix}")
                sem = SEMImage(path, pixel_size_nm=12.5)
                self.assertEqual(sem.load().dtype, dtype)
                mask = sem.segment(SegmentationOptions(threshold=0.5, crop=(2, 1, 6, 5), min_solid_pixels=0, fill_holes_pixels=0))
                self.assertEqual(mask.shape, (5, 6))
                self.assertTrue(np.all(~mask[1:4, 1:4]))
                self.assertAlmostEqual(sem.metadata["width_mm"], 6 * 12.5e-6)
                self.assertAlmostEqual(sem.metadata["height_mm"], 5 * 12.5e-6)
                self.assertEqual(sem.metadata["crop"], (2, 1, 6, 5))

    def test_cleanup_and_component_selection(self):
        pixels = np.zeros((12, 12), dtype=np.uint8)
        pixels[1:6, 1:6] = 255
        pixels[3, 3] = 0  # one-pixel pore
        pixels[8:10, 8:10] = 255  # separate four-pixel solid
        pixels[10, 1] = 255  # solid speck
        sem = SEMImage(self.write_image(pixels), pixel_size_nm=10)
        mask = sem.segment(SegmentationOptions(min_solid_pixels=2, fill_holes_pixels=1, component="all"))
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[10, 1])
        self.assertTrue(mask[8, 8])
        self.assertEqual(sem.metadata["components_retained"], 2)
        self.assertEqual(sem.metadata["changed_pixels"], 2)
        geometry = sem.create_geometry(tolerance_px=0)
        self.assertEqual(len(geometry), 2)
        mask = sem.segment(SegmentationOptions(min_solid_pixels=0, fill_holes_pixels=0, component="largest"))
        self.assertIsNone(sem.geometry)
        self.assertFalse(mask[8, 8])
        self.assertEqual(sem.metadata["components_retained"], 1)

    def test_percentile_threshold_adapts_to_brightness_and_crop(self):
        gradient = np.tile(np.arange(20, dtype=np.uint16), (20, 1))
        masks = []
        cutoffs = []
        for name, pixels in (("dim", 1000 + gradient * 1000),
                             ("bright", 10000 + gradient * 2500)):
            sem = SEMImage(self.write_image(pixels, f"{name}.tif"), pixel_size_nm=10)
            mask = sem.segment(SegmentationOptions(threshold_percentile=50,
                                                    min_solid_pixels=0, fill_holes_pixels=0))
            masks.append(mask)
            cutoffs.append(sem.metadata["threshold_raw"])
            self.assertEqual(sem.metadata["threshold_mode"], "percentile")
            self.assertEqual(sem.metadata["threshold_percentile"], 50)
            self.assertEqual(sem.metadata["original_solid_pixels"], 200)
        np.testing.assert_array_equal(masks[0], masks[1])
        self.assertEqual(int(masks[0].sum()), 200)
        self.assertNotEqual(cutoffs[0], cutoffs[1])
        sem = SEMImage(self.root / "dim.tif", pixel_size_nm=10)
        cropped = sem.segment(SegmentationOptions(threshold_percentile=50, crop=(0, 0, 10, 20),
                                                   min_solid_pixels=0, fill_holes_pixels=0))
        self.assertEqual(cropped.shape, (20, 10))
        self.assertEqual(int(cropped.sum()), 100)

    def test_full_image_hole_and_stage_reset(self):
        pixels = np.full((32, 32), 255, dtype=np.uint8)
        pixels[10:16, 10:16] = 0
        sem = SEMImage(self.write_image(pixels), pixel_size_nm=10)
        result = sem.run(segmentation=SegmentationOptions(min_solid_pixels=0, fill_holes_pixels=0))
        self.assertEqual(sem.metadata["crop"], (0, 0, 32, 32))
        self.assertEqual(len(result.contours), 1)
        self.assertEqual(len(result.contours[0].interiors), 1)
        self.assertTrue(np.all(result.points >= -1e-12))
        self.assertLessEqual(float(result.points[:, 0].max()), 32 * 10e-6 + 1e-12)
        self.assertLessEqual(float(result.points[:, 1].max()), 32 * 10e-6 + 1e-12)
        centers = result.points[result.triangles].mean(axis=1)
        # Image rows increase downwards, so the pore at rows 10:16 lies at
        # mesh Y coordinates 16:22 pixels above the lower edge.
        self.assertFalse(np.any((11e-5 < centers[:, 0]) & (centers[:, 0] < 15e-5)
                                & (17e-5 < centers[:, 1]) & (centers[:, 1] < 21e-5)))
        sem.create_geometry(tolerance_px=0)
        self.assertIsNone(sem.mesh)
        sem.segment(SegmentationOptions(min_solid_pixels=0, fill_holes_pixels=0))
        self.assertIsNone(sem.geometry)
        sem.load()
        self.assertIsNone(sem.mask)

    def test_msh_reopens_and_matches_returned_arrays(self):
        import gmsh

        pixels = np.full((24, 24), 255, dtype=np.uint8)
        pixels[8:13, 8:13] = 0
        sem = SEMImage(self.write_image(pixels), pixel_size_nm=20)
        result = sem.run(segmentation=SegmentationOptions(min_solid_pixels=0, fill_holes_pixels=0),
                         meshing=MeshingOptions(boundary_size_nm=40, interior_size_nm=80))
        path = result.save_msh(self.root / "mesh.msh")
        self.assertGreater(path.stat().st_size, 0)
        self.assertEqual(result.quality["triangles"], len(result.triangles))
        self.assertGreater(result.quality["minimum_quality"], 0)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(path))
            tags, coords, _ = gmsh.model.mesh.getNodes()
            reopened_points = np.asarray(coords).reshape(-1, 3)[:, :2]
            _, cells = gmsh.model.mesh.getElementsByType(2)
            node_index = {int(tag): i for i, tag in enumerate(tags)}
            reopened_triangles = np.array([node_index[int(tag)] for tag in cells]).reshape(-1, 3)
            np.testing.assert_allclose(reopened_points, result.points)
            np.testing.assert_array_equal(reopened_triangles, result.triangles)
            self.assertEqual(len(reopened_triangles), result.quality["triangles"])
        finally:
            gmsh.finalize()

    def test_invalid_inputs_and_required_stages(self):
        with self.assertRaisesRegex(ValueError, "pixel_size_nm"):
            SEMImage(self.root / "missing.png", pixel_size_nm=0)
        color = self.write_image(np.zeros((4, 4, 3), dtype=np.uint8), "color.png")
        with self.assertRaisesRegex(ValueError, "single-channel"):
            SEMImage(color, pixel_size_nm=10).load()
        pixels = np.full((8, 8), 255, dtype=np.uint8)
        sem = SEMImage(self.write_image(pixels), pixel_size_nm=10)
        with self.assertRaisesRegex(RuntimeError, "segment"):
            sem.create_geometry()
        with self.assertRaisesRegex(ValueError, "crop"):
            sem.segment(SegmentationOptions(crop=(7, 0, 2, 2)))
        with self.assertRaisesRegex(ValueError, "threshold"):
            sem.segment(SegmentationOptions(threshold=float("nan")))
        for invalid in (-1, 0, 100, 101, float("nan"), True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "threshold_percentile"):
                    sem.segment(SegmentationOptions(threshold_percentile=invalid))
        with self.assertRaisesRegex(ValueError, "varying image intensities"):
            sem.segment(SegmentationOptions(threshold_percentile=50))
        with self.assertRaisesRegex(ValueError, "no solid"):
            sem.segment(SegmentationOptions(threshold=1, min_solid_pixels=100))
        sem.segment(SegmentationOptions(min_solid_pixels=0, fill_holes_pixels=0))
        with self.assertRaisesRegex(ValueError, "tolerance_px"):
            sem.create_geometry(tolerance_px=3)
        sem.create_geometry()
        with self.assertRaisesRegex(ValueError, "boundary_size_nm"):
            sem.generate_mesh(MeshingOptions(boundary_size_nm=100, interior_size_nm=50))

    def test_import_has_no_gui_or_physics_dependency(self):
        process = subprocess.run(
            [sys.executable, "-c", "import sys; import sem2msh; assert 'gmsh' not in sys.modules; assert 'dolfinx' not in sys.modules; assert 'sem_meshing_gui' not in sys.modules"],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
