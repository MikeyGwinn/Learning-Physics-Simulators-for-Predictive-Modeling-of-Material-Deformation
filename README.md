# SEM2msh

`sem2msh` converts a single grayscale SEM image into a calibrated 2D Gmsh
triangle mesh. It does not run physics or require the PhaseFieldTest GUI.

Install with `python -m pip install .` in a Python 3.10+ environment with the
system libraries required by Gmsh. Import the API from `sem2msh`:

```python
from sem2msh import SEMImage, SegmentationOptions, MeshingOptions

image = SEMImage("micrograph.tif", pixel_size_nm=10)
result = image.run(
    segmentation=SegmentationOptions(
        threshold=0.45,
        crop=(1400, 900, 384, 384),  # optional: left, top, width, height
        min_solid_pixels=16,
        fill_holes_pixels=8,
        component="all",
    ),
    tolerance_px=0.5,
    meshing=MeshingOptions(
        boundary_size_nm=30,
        interior_size_nm=80,
        refinement_distance_nm=100,
    ),
)
result.save_msh("specimen.msh")

print(result.points.shape, result.triangles.shape, result.quality)
```

For inspection between stages, call `image.load()`, `image.segment(...)`,
`image.create_geometry(...)`, and `image.generate_mesh(...)` separately.
`image.image` is the original integer array; `image.mask` is the cropped solid
mask; `image.geometry` contains ROI-local Shapely polygons. The result has
`mask`, `contours`, `points`, `triangles`, `parameters`, `quality`, and `metadata`.
The arrays use zero-based triangle indices. Calling an earlier stage clears its
later results.

Only single-channel 8- and 16-bit images are accepted. Intensities are divided
by the original dtype maximum before thresholding; values **below** the
threshold are void. Crop coordinates use the top-left image origin. Contours
use local crop pixel coordinates with Y upwards; mesh points use millimeters
with the lower-left crop corner at (0, 0). Pixel size is required and assumed
equal on both axes. The Gmsh output contains physical surface group 1.

For images whose overall brightness varies, set
`SegmentationOptions(threshold_percentile=30)` to choose the cutoff separately
from each cropped image's raw intensity distribution. This selects approximately
the brightest 70% of pixels as solid before cleanup; the fixed `threshold`
value is ignored in this mode. The actual normalized and raw cutoffs are saved
in `result.metadata["threshold"]` and `result.metadata["threshold_raw"]`.
Percentiles assume a similar solid fraction across images, so inspect masks
before treating the resulting geometry as a material measurement.

Run the focused tests with `python -m unittest discover -s tests -v`.

For the distributed Skärberg image conversion benchmark, see
[benchmark/README.md](benchmark/README.md).
