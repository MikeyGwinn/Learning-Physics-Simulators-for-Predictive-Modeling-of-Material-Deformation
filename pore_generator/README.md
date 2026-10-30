# Synthetic pore generator

Generate reproducible circular pores in a square sample and export Gmsh geometry
as tagged material regions or a perforated specimen. Python 3.12 or newer is
required. Generation uses the standard library; meshing requires Gmsh 4.15.2.

## Directory contents

| Path | Purpose |
| --- | --- |
| `generation.py` | Configuration, radius sampling, placement, and porosity statistics |
| `export.py` | JSON/CSV output, Gmsh geometry, and SVG previews |
| `meshing.py` | Gmsh mesh generation, inspection, and solver metadata |
| `__main__.py` | Command-line interface |
| `generate_batch.py` | Generate complete samples using consecutive seeds and record rejected seeds |
| `mesh_batch.py` | Mesh successful batch samples and verify completed outputs before skipping them |
| `examples/` | Example JSON configuration |
| `tests/` | Generator and mesh correctness checks |
| `requirements-mesh.txt` | Optional meshing dependency |
| `results/` | Generated samples and meshes; ignored by Git |

## Usage

Run these commands from the repository root:

```bash
python3 -m venv pore_generator/.venv
source pore_generator/.venv/bin/activate
python -m pip install -r pore_generator/requirements-mesh.txt
python -m pore_generator generate --seed 42 --output pore_generator/results/seed-42
python -m pore_generator mesh --sample pore_generator/results/seed-42 --format both
```

Use `--format material` (the default) or `--format perforated` to mesh one
representation. Each selected mesh is generated in sequence. Existing samples,
mesh results, and modified geometry files are not overwritten. Some Linux
installations also require the system `libGLU` library.

To use a configuration file:

```bash
python -m pore_generator generate --seed 42 --config pore_generator/examples/pore_config.json --output pore_generator/results/config-seed-42
python -m pore_generator generate --help
```

Individual flags override JSON values, which override the defaults. Flags use
hyphens, for example `--target-porosity 0.15`.

## Batch generation

```bash
python -m pore_generator.generate_batch --count 500 --start-seed 0 --output pore_generator/results/batch-500
```

`--count` counts successful samples, not pores, and defaults to 500. Seeds start
at `--start-seed` (default 0) and advance after every attempt. Use `--config`
with the same JSON keys as single-sample generation. Gmsh is not required.

The output must be a new directory. Successful samples are saved as
`seed-000000/`, `seed-000001/`, and so on, each containing the standard sample
outputs and both `.geo` representations. Rejected seeds leave gaps in these
directory names. Parameters remain fixed across attempts.

`batch.json` records the configuration, Python version, requested count, starting
seed, run status, and totals. `samples.csv` records every attempted seed, its
status, relative sample directory, pore counts, achieved porosity, and any error.
The CSV is flushed after each attempt. Filesystem and unexpected errors stop
generation and mark the batch failed.

The dataset contains successful placements; rejected seeds and their reasons
remain in the CSV. There is no limit on how many seeds are attempted, so a
configuration that cannot succeed will keep running until interrupted. Ctrl+C
retains completed samples and records, marks the batch interrupted, and exits
with status 130. Existing batches cannot be resumed.

## Batch meshing

```bash
python -m pore_generator.mesh_batch --batch pore_generator/results/batch-500 --format perforated
```

Gmsh 4.15.2 must be installed in the active environment. `--format` accepts
`perforated` (the default), `material`, or `both`. The script reads successful
entries from `samples.csv`, including samples retained after interrupted
generation, and meshes the selected representations sequentially.

Rerunning skips completed outputs after checking the saved geometry, geometry
hash, mesh inspection results, summary, and perforated compatibility metadata.
Incomplete or inconsistent outputs are preserved and recorded as failures that
require manual inspection and cleanup. Mesh summaries created without a geometry
hash cannot be skipped. No existing mesh is deleted or overwritten.

`meshing.csv` contains the latest invocation's seed, relative sample directory,
representation, status (`generated`, `skipped`, or `failed`), and error. Rows are
flushed after each representation. Individual mesh failures are recorded while
processing continues; missing dependencies and filesystem errors stop the run.
Any mesh failure produces a nonzero exit status. Ctrl+C exits with status 130
and retains existing outputs and CSV rows.

## Parameters and modeling assumptions

| JSON key | Default | Meaning |
| --- | --- | --- |
| `size_mm` | 1.0 | Side length of the square |
| `radius_mu` | 2.941 | Mean of the natural logarithm of radius in micrometers |
| `radius_sigma` | 0.703 | Standard deviation of that logarithm |
| `min_radius_um` | 1.0 | Lower radius cutoff; smaller draws are resampled |
| `geometric_radius_um` | 20.0 | Minimum radius represented geometrically |
| `target_porosity` | 0.20 | Target area fraction, including small pores |
| `clearance_um` | 0.0 | Minimum pore-to-pore and pore-to-boundary gap |
| `mesh_size_um` | 4.0 | Target triangle size |

Defaults use the radius distribution in Chakraborty, Sabharwall, and Carroll
(2016), *A phase-field approach to model multi-axial and microstructure dependent
fracture in nuclear grade graphite*, Section 3, Figure 2, and Table 2:
[doi:10.1016/j.jnucmat.2016.04.006](https://doi.org/10.1016/j.jnucmat.2016.04.006).
The distribution is `ln(radius / micrometer) ~ N(2.941, 0.703²)`, truncated below
1 micrometer. These parameters describe radius, not diameter.

Output coordinates and radii are in **millimeters**, with a bottom-left origin,
x pointing right, and y pointing up. The SVG previews flip y for display.

Pore centers are sampled uniformly within the permitted interior. Circles stay
strictly inside the square and do not overlap or touch. Each radius gets up to
10,000 placement attempts; an unplaceable radius aborts generation rather than
being replaced or resized. Sampling above the lower cutoff also has a
10,000-draw limit. Some seeds can fail even at the default porosity.

Generation stops when total pore area first reaches or exceeds the target.
Actual porosity and overshoot are recorded. These placement and stopping rules
can affect the empirical radius distribution and are documented modeling choices.
A seed reproduces a sample for the same configuration and Python implementation
and version, recorded in `sample.json`.

Pores below the geometric cutoff are saved and counted in total porosity, but
occupy matrix material in both meshes. No small-pore stiffness correction is
applied. Material-region meshes contain tagged pore cells; perforated meshes
remove those cells as holes. The exporter assigns no material properties.

## Outputs

Each sample directory contains:

- `sample.json`: configuration, seed, coordinate conventions, pore table, and statistics.
- `pores.csv`: `id,x_mm,y_mm,radius_mm,kind`, with kind `explicit` or `small`.
- `preview.svg`: all pores, with explicit pores blue and small pores orange.
- `material_mesh/specimen.geo`: matrix and pore regions sharing circle boundaries.
- `perforated_mesh/specimen.geo`: the same explicit pores represented as holes.

Meshing adds `specimen.msh`, `mesh_summary.json`, and `mesh_preview.svg` to each
selected mesh directory. Meshes use ASCII MSH 4.1 and first-order triangles.
The summary records the geometry hash and reports cell counts, areas, quality,
connectivity, boundary counts, and pore-area error from approximating circles
with straight edges.

| Representation | Surface tags | Curve tags |
| --- | --- | --- |
| Material regions | matrix=1, pores=2 | left=1, right=2, bottom=3, top=4, interfaces=5 |
| Perforated | matrix=1 | left=1, right=2, bottom=3, top=4, hole edges=5 |

Pore-related groups are omitted when there are no explicit pores. Inspection
checks triangle areas and quality, tags, grip locations, matrix connectivity,
area coverage, and pore-edge adjacency, both before and after saving the mesh.

## Solver handoff

`perforated_mesh/geometry.json` contains the dimensions, geometry hash, porosity
fractions, mesh size, and readiness flag expected by John's SEMtoMesh solver.
`porosity` is the meshed hole fraction; analytic explicit, small, and total sample
porosities are recorded separately. The geometry hash is SHA256 of the `.geo`
file, including mesh settings. Readiness indicates that mesh checks passed.

From a SEMtoMesh checkout with its DOLFINx environment active:

```bash
python simulate_fracture.py --image-mesh /absolute/path/to/perforated_mesh --config /absolute/path/to/fracture_config.json --output results/synthetic-fracture --backend cpu
```

Set `geometry.width` and `geometry.height` in the solver configuration to the
sample size (1.0 mm by default), and `mesh.size` to the requested mesh size
(0.004 mm by default). Material parameters, thickness, fracture length scale,
and loading are supplied by the solver configuration.

## Correctness checks

```bash
python -m unittest discover -s pore_generator/tests -v
git diff --check
```

Generator checks use the standard library. The mesh check covers both
representations, a solid square, compatibility metadata, and overwrite
protection. It is skipped when Gmsh is unavailable.
