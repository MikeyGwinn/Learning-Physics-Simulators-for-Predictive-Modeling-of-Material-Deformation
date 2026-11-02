# Learning Physics Simulators for Predictive Modeling of Material Deformation

[![CI](https://github.com/MikeyGwinn/Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation/actions/workflows/ci.yml/badge.svg)](https://github.com/MikeyGwinn/Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation/actions/workflows/ci.yml)

An Oregon State University capstone research project, supervised by Dr. Tianyi Chen and Dr. Chanho Kim.

## Background

Materials don't fail all at once. Under load, small regions deform and microcracks form, grow, and join until the part breaks. In-situ scanning electron microscopy (SEM) lets researchers watch this happen in a sample as it is loaded. Physics-based simulations of the process, such as phase-field fracture models, can take hours of compute per run.

This project asks whether a **neural physics simulator** can learn these dynamics instead. The model would be a particle-based network that looks at a material's current state and recent deformation, then predicts how it will deform next. We train it on phase-field fracture simulations of porous nuclear graphite and check it against Dr. Chen's SEM tests of ET-10 graphite. The long-term motivation is a digital twin that predicts how damage evolves in components such as those in nuclear reactors.

## How the pipeline fits together

```text
pore layouts / SEM images ──► 2D mesh ──► phase-field fracture ──► particle ──► neural ──► predicted
  (synthetic or real)         (Gmsh)      simulation (MOOSE)       trajectories  simulator   deformation
```

| Stage | What it does | Status |
| --- | --- | --- |
| Geometry and meshing | Builds 2D graphite specimens from synthetic pore layouts or from SEM images, and meshes them with Gmsh | In progress |
| Simulation | Runs the phase-field fracture model of Chakraborty et al. (2016) in MOOSE on the OSU HPC (`scripts/moose_container.sbatch`) | Running on the HPC; benchmark validation in progress |
| Export | Converts the simulated displacements into particle trajectories the model can read | Planned (see `docs/rfcs/`) |
| Model | Adapts Dr. Kim's particle dynamics model to predict 2D deformation 10 load steps ahead | Baseline runs on the HPC (`modeling/HPC_setup.md`); adaptation in progress |
| Evaluation | Compares predictions with simple extrapolation baselines and with real SEM tests | Planned |

[docs/requirements.md](docs/requirements.md) has the goals and success metric. [docs/design.md](docs/design.md) has the design. [docs/risks.md](docs/risks.md) has the known risks.

## Quickstart

This sets up a local environment and runs the test suite that CI runs. The simulation and model stages need an OSU HPC account (see below). You need Git and Python 3.12 or newer. On Linux, Gmsh also needs `libGLU` (on Ubuntu: `sudo apt-get install libglu1-mesa`).

```bash
git clone https://github.com/MikeyGwinn/Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation.git
cd Learning-Physics-Simulators-for-Predictive-Modeling-of-Material-Deformation
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r pore_generator/requirements-mesh.txt
python -m unittest discover -s pore_generator/tests -v
```

All tests should pass.

## Running on the OSU HPC

- **MOOSE simulations:** `sbatch scripts/moose_container.sbatch path/to/input.i` runs an input file inside INL's MOOSE container. The setup log is in `logs/`.
- **Model training:** follow [modeling/HPC_setup.md](modeling/HPC_setup.md) to set up and train Dr. Kim's particle dynamics model on a GPU node.

## Repository layout

| Path | Contents |
| --- | --- |
| `docs/` | Requirements, design, risk register, and RFCs for past decisions (`docs/rfcs/`) |
| `scripts/` | HPC job scripts |
| `logs/` | HPC setup logs |
| `modeling/` | Model setup guides |
| `.github/workflows/` | CI |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. AI coding agents should read [AGENTS.md](AGENTS.md).
