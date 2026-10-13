# SEM meshing GUI and image-to-fracture benchmark

## Start the GUI

```bash
cd /nfs/hpc/share/scumnioj/PhaseFieldTest
source activate.sh
python sem_meshing_gui.py --port 8765
```

The launcher automatically stops an existing process using the selected TCP
port before starting the GUI, so rerunning this command replaces an older GUI
instance.

Open <http://127.0.0.1:8765>. On a remote HPC node, forward the port from
your workstation and then open the same URL locally:

```bash
ssh -L 8765:127.0.0.1:8765 YOUR_USER@THE_NODE_RUNNING_THE_GUI
```

The server binds only to localhost. It uses Python's standard HTTP server,
Pillow, NumPy, SciPy, scikit-image, Shapely and Gmsh. All are available in
the workspace `.conda` environment; Shapely was added without changing the
DOLFINx/PETSc packages. No cloud account or API key is needed.

## Dataset and provenance

Data: Skärberg et al., *Convolutional neural networks for segmentation of
FIB-SEM nanotomography data from porous polymer films for controlled drug
release*, DOI [10.5281/zenodo.4317170](https://zenodo.org/records/4317170),
CC BY 4.0. The published voxel spacing is 10 × 10 × 50 nm. GUI coordinates
are displayed in micrometers; Gmsh/FEM exports use millimeters.

Six actual raw/segmentation pairs were fetched: slices 0 and 100 from HPC22,
HPC30 and HPC45. Raw images are 3372 × 2247 pixels, 16-bit TIFF. These are
spatial slices of a static 3D specimen, not time frames of deformation.
The supplied PNG masks are **author CNN predictions**, not expert manual
ground truth. Upstream `postprocessing.m` defines porosity as `mean(~A)`
and writes `A` to PNG: white/true is **solid**. Raw TIFF numbering starts
at 1; PNG numbering starts at 0. The downloader explicitly pairs them.

`fetch_skarberg.py` uses HTTP byte ranges to fetch individual ZIP members
without downloading the 9.2 GB archive. Each extracted ZIP member is
validated by its CRC and receives a SHA256 in `manifest.json`. The upstream
archive MD5 is recorded as a release identifier; partial downloads cannot
verify the whole-archive MD5. Relevant author preprocessing/export scripts
are retained under `datasets/skarberg/upstream_code/` for provenance and
are not executed by this application.

Fetch more slices and click **Refresh** in the GUI:

You can also choose a slice index (0–199) and click **Fetch slice** inside
the GUI. It fetches one raw/segmentation pair for the selected sample and
switches to that image. The 600 source slices can be browsed incrementally;
the complete archive is never required. Cached files are checked against
upstream ZIP CRCs, and manifest updates are atomic.

```bash
python fetch_skarberg.py --samples 22 30 45 --indices 25 50 75 125 150 175 199
```

An existing archive can be read with `--archive /path/to/fib_sem_cnn.zip`.
Downloading data is separate from benchmark timing.

## Controls and inspection

- Choose an image, or use Previous/Next. Crop origin is the upper-left pixel;
  crop dimensions are pixels. Physical y is flipped for conventional FEM
  coordinates. Image switching clears the previous mesh.
- The GUI segments exclusively from a raw intensity threshold. It uses the
  original TIFF dtype range [0,65535], normalized to [0,1], not the displayed
  image contrast. The colored histogram behind the slider shows the current
  ROI distribution. Author masks remain available to scripted workflows.
- Preserve all solid components by default. Selecting the largest component
  is an explicit geometric alteration. Removed/changed pixels and resulting
  porosity are reported; isolated material is not silently joined to a load path.
- Solid speck removal defaults to 16 px² and small-pore filling to 8 px². Their effects
  are included in the changed-pixel audit.
- Contour tolerance controls simplification in pixels, with topology preserved.
  Marching-squares boundary representation has small area differences from
  the binary pixel-cell area; the geometry report quantifies this difference.
- Pore-edge size, interior size and refinement distance are nanometers.
  Gmsh's distance/threshold sizing field refines boundaries and transitions
  toward the requested interior size. Smoothing and algorithm are adjustable.
- The fracture length scale is exported for later simulation; the GUI shows
  its ratio to requested interior size. Meshing does not automatically change
  the physical length scale. Requested sizes and quality statistics do not
  establish mesh convergence.
- Wheel to zoom, drag to pan, and use Fit image/Fit mesh. Toggle solid-mask
  overlay, SEM backdrop, and mesh edges. Meshes use WebGL2 vertex/index buffers
  with GPU transforms, falling back to a cached Canvas 2D path when WebGL2 is
  unavailable. Input is coalesced to animation frames and backing stores resize
  only when the layout changes.

Builds run in an isolated subprocess, leaving the UI responsive. Changing
settings during a build prevents an old result from being shown as current.
Builds are serialized and have a five-minute timeout. Invalid crops, missing
solid material and geometry errors appear in the status panel.

Exports are stored under `results/sem_gui/<job-id>/`:

- `parameters.json`: reproducible image/crop/mesh settings.
- `mesh/specimen.msh`: physical, first-order triangular mesh.
- `mesh/solid_mask.png`, `image.png`, `contours.json`: inspected input geometry.
- `mesh/geometry.json`: input hashes, calibration, component changes, pore
  area, contour fidelity, node/element/DOF counts and element quality.
- `mesh/mesh.npz`: browser rendering coordinates/connectivity.
- `fracture_config.json`: illustrative simulation parameters.
- `job.json` and `build.log`: build outcome and diagnostic evidence.

Boundary tags are left=1, right=2, bottom=3, top=4 and pore/other=5.
The mesher also supports pores intersecting crop edges and nested solid
islands. A fracture-ready mesh must have one connected component spanning
the left and right edges and actual tagged grip facets. Multiple components
are allowed for geometry inspection, but the present fracture driver rejects
floating/disconnected material rather than supplying artificial constraints.

## Run and animate fracture from the GUI

On heterogeneous HPC nodes, generated FEM modules use the host system GCC
when available and a compilation cache isolated by hostname, glibc, Python
and compiler. This prevents reuse of modules linked against another node's
newer glibc. The first simulation on each node requires fresh form compilation.

When a mesh has one connected solid component spanning both grips, **Simulate
fracture** becomes available. The CPU/GPU selector changes the PETSc backend;
the generated `specimen.msh` is passed directly to `simulate_fracture.py
--image-mesh`, bypassing synthetic geometry generation. Configuration
dimensions and the geometry hash remain tied to the selected mesh job.

The solve runs asynchronously and reports accepted load steps, separation and
maximum damage. Every accepted field frame is saved. After convergence,
`visualize.py` renders the displaced mesh and damage field together with the
force curve, and the GUI embeds the resulting MP4 with playback and download
controls. Deformation scale changes only the animation; it does not change the
finite-element solution. Videos last at least five seconds; short runs play
their accepted frames more slowly while longer runs retain the selected FPS.
The vertical rigid-translation gauge is placed at an
existing left-grip vertex, while top, bottom and pore boundaries remain free.

If the solver reaches its minimum load increment or another convergence limit,
the GUI still renders the saved accepted steps. The player and movie label this
as a partial result and show the stopping reason. Failed trial states are never
included. Runs that fail before accepting any load step cannot produce an
animation. The solver status remains failed; the animation does not imply
completion of the requested load path. For existing failed runs, use
`python visualize.py RUN_FOLDER --allow-partial`.

Simulation artifacts are stored in `results/sem_gui/<simulation-id>/run/`.
Material properties, thickness, the 2% nominal strain and crop boundary
interpretation remain illustrative until calibrated to the experiment.

## Full image → mesh → fracture benchmark (not executed)

Validate the plan without processing images, building meshes or solving:

```bash
python image_workflow_benchmark.py --dry-run
```

The default plan is `configs/sem_workflow_benchmark.json`: two images,
three interior mesh sizes (80,60,40 nm), boundary size half the interior
size, CPU and GPU, one warmup and five measured repetitions. This schedules
72 independent processes, including 60 measured runs. It explicitly retains
the largest component, with the removal audited for each image. Physical
calibration, contour tolerance, ROI, material parameters, fracture length
and nominal loading strain stay fixed across mesh refinement/backends.

Execution requires an explicit flag. This command is provided for future use
and **was not run during implementation**:

```bash
python image_workflow_benchmark.py --execute --output results/image_workflow_benchmark
python plot_image_workflow.py results/image_workflow_benchmark
```

Each independent worker decodes the real raw/mask files, segments/cleans,
extracts contours, builds a mesh, imports that exact mesh into DOLFINx,
constructs the fracture problem and completes the configured load path.
Both backends use matched GMRES/GAMG and tolerances. FEM assembly and
image processing remain CPU operations. nm-scale configs use smaller
absolute solver tolerances to avoid accepting the mm-scale demo's residual
floor on much smaller damage equations. GPU selection uses CUDA PETSc and
requires an allocated NVIDIA GPU.

The primary end-to-end metric is parent-measured `process_wall_seconds`,
including process startup, library imports, image decode, segmentation,
contours, meshing, FEM import/form compilation/setup, solves and final outputs.
Separately recorded stage times explain costs. Form-compilation disk caches
can become warm after warmup runs; each repetition still starts a new process
and initializes its own device context. Download/network time is excluded.
OpenMP/BLAS thread counts are fixed; hardware/device visibility is recorded.

`measurements.csv` preserves warmups (negative repeat) and failures. Successful
GPU results are checked against CPU displacement, damage, reaction curves
and stored energy. Mismatches, timeouts and unconverged runs are failures and
cause a nonzero benchmark exit. Actual displacement+damage DOFs are recorded,
not inferred from requested mesh sizes. Geometry/source hashes document
identical inputs. Energy imbalance is reported for scientific assessment.

Plotting excludes warmups and failed results and produces per-image CPU/GPU
wall time versus DOFs, speedup, and mean stage-cost figures. Stage charts
include startup/output/other cost so they account for the measured total.

`configs/sem_fracture.json` uses illustrative E, Gc, thickness and loading.
These are not measured properties of the Skärberg polymers. A 2D cropped
tomography slice also does not establish actual free surfaces or grip
boundary conditions. Calibrate material parameters, select suitable 2D/3D
mechanics, and check fracture/energy convergence before quantitative use.

Verification scripts under `tests/` were removed during cleanup. They are not
runtime dependencies. `image_workflow_benchmark.py --dry-run` still validates
the SEM benchmark plan without running the benchmark.
