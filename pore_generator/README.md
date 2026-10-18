# Synthetic pore generator

Checkpoint 1 implements synthetic pore generation, saved data, and an SVG preview.
It uses Python 3.12 or newer and only the Python standard library. Run commands
from the repository root; no package installation is needed.

```bash
python3 -m venv pore_generator/.venv
source pore_generator/.venv/bin/activate
python -m pore_generator generate --seed 42 --output pore_generator/results/seed-42
```

Open `pore_generator/results/seed-42/preview.svg` in a browser. Blue circles are explicit pores;
orange circles are small pores. Output also includes:

- `sample.json`: schema version, seed, configuration, coordinates, every pore,
  achieved porosity by class, overshoot, and modeling assumptions.
- `pores.csv`: `id,x_mm,y_mm,radius_mm,kind`, where kind is `explicit` or `small`.

An existing output directory is rejected. Sampling or placement failure exits
with an error before creating an output directory. Seeds reproduce samples for
the same configuration and Python implementation/version, which are recorded in
`sample.json`. The saved pore table is the authoritative
realization for later exports.

### Parameters and conventions

```bash
python -m pore_generator generate --seed 42 --config pore_generator/examples/pore_config.json --output pore_generator/results/config-seed-42
python -m pore_generator generate --help
```

JSON keys match the following table. CLI flags use hyphens, for example
`--target-porosity 0.15`. Precedence is defaults, then JSON, then individual flags.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `size_mm` | 1.0 | Side length of the square |
| `radius_mu` | 2.941 | Mean of the natural logarithm of radius in micrometers |
| `radius_sigma` | 0.703 | Standard deviation of that logarithm |
| `min_radius_um` | 1.0 | Lower truncation cutoff; smaller draws are resampled |
| `geometric_radius_um` | 20.0 | Radii at or above this threshold are explicit |
| `target_porosity` | 0.20 | Target total area fraction, including small pores |
| `clearance_um` | 0.0 | Minimum pore-to-pore and pore-to-boundary gap |
| `mesh_size_um` | 4.0 | Reserved target mesh size for the later mesh exporter |

Coordinates and radii in output records are **millimeters**. The origin is at the
bottom-left, x points right, and y points up. The SVG flips y only for display.
The radius sampler implements `ln(radius / micrometer) ~ N(2.941, 0.703²)`;
these are log-space parameters for **radius**, not diameter.

The parameter source is Chakraborty, Sabharwall, and Carroll (2016),
*A phase-field approach to model multi-axial and microstructure dependent fracture
in nuclear grade graphite*, Section 3, Figure 2, and Table 2:
<https://doi.org/10.1016/j.jnucmat.2016.04.006>.

### Scientific assumptions and current limits

Circles remain strictly inside the square and do not overlap or touch. For each
sampled radius, centers are drawn uniformly in the permitted interior until one
fits. A radius that cannot fit aborts generation; failed placement is not replaced
with a smaller radius. Placement is limited to 10,000 attempts per pore. Radius
sampling below the cutoff is also limited to 10,000 draws to reject impractical
parameter choices.

Some seeds can fail even at the default target porosity: previously placed pores
may leave no room for a later large radius. For example, seed 7 fails placement
under the defaults, while seed 42 succeeds. This follows the fixed-radius policy;
generation does not silently discard difficult pores. Use another seed or an
explicitly changed configuration if a sample fails.

Generation stops at the first total pore-area crossing of the target. Radii are
not rescaled to force an exact porosity. Actual porosity and overshoot are reported;
small pores contribute to the total. Placement, clearance, cutoff, and stopping
rules can affect the empirical distribution. These construction rules are
provisional modeling choices, not a complete reproduction of the paper.

All small pores are saved, but their effective stiffness correction is not applied.
Mesh generation is **not implemented yet**. Subsequent review checkpoints will add
`material_mesh/` with tagged matrix/pore regions, `perforated_mesh/` with large-pore
holes, and then a small-pore porosity grid. Gmsh will be needed only for meshing.
There is no fracture solve or claim of validated material behavior at this stage.

### Development and checks

This directory contains the generation, export, and CLI modules, along with its
own `examples/`, `tests/`, and ignored `results/`. Commands below run from the
repository root. Generated results, local environments, and caches are ignored
by this directory's `.gitignore`.

```bash
python -m unittest discover -s pore_generator/tests -v
git diff --check
```

