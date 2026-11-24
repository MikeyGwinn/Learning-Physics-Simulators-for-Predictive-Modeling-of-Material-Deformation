# AGENTS.md

Instructions for AI coding agents working in this repository. This file points at the living docs instead of repeating them; when this file and a living doc disagree, the living doc wins. Fix this file in the same PR. Any other agent instruction file (for example `CLAUDE.md`) should contain only a pointer to this one.

## Project in one paragraph

This is a research capstone (CS 461, Oregon State University) supervised by Dr. Tianyi Chen and Dr. Chanho Kim. We generate 2D phase-field fracture simulations of porous graphite in MOOSE, export them as particle trajectories, and train a particle-based neural simulator to predict how a sample deforms several load steps ahead. We then check the predictions against Dr. Chen's in-situ SEM tests of ET-10 graphite. A SiC/SiC composite dataset is expected later.

## Read these first

| Need | Read |
| --- | --- |
| What we are building, constraints, non-goals, acceptance criteria | `docs/requirements.md` |
| How the system fits together, data format v0, interfaces, deployment target | `docs/design.md` |
| Known risks and external waits (read-only for agents, see below) | `docs/risks.md` |
| Setup and quickstart from a clean clone | `README.md` |
| Branching, commit messages, PR checklist, Definition of Done | `CONTRIBUTING.md` |
| Why past technical decisions were made | `docs/rfcs/` |

## Components

Each component has its own README with its setup and test commands. Use those commands; don't invent new ones. If a path below is missing at your commit, that component hasn't been merged yet. Ask instead of recreating it.

| Component | Path | Test command (from the repo root) |
| --- | --- | --- |
| Synthetic pore generator and Gmsh meshing | `pore_generator/` | `python -m unittest discover -s pore_generator/tests -v` |
| SEM image to Gmsh mesh (`sem2msh`) | `sem2msh/`, `benchmark/` | `python -m unittest discover -s tests -v` (after `python -m pip install .`) |
| MOOSE on the OSU HPC (Apptainer container) | `scripts/moose_container.sbatch`, `logs/` | none (runs on the HPC only) |
| Particle dynamics model (adapted from Dr. Kim's code) | not in this repo yet | none yet |

Python 3.10+ is required (3.12+ for `pore_generator`). Gmsh is pinned in `pore_generator/requirements-mesh.txt`. There is no shared environment file, linter, or formatter yet. Match the style of the code around you and don't add tooling in an unrelated PR.

## Scientific guardrails

- Units and coordinates are part of every interface. `sem2msh` mesh points are in millimeters with the origin at the lower-left corner of the crop and Y pointing up. Image contours are in pixels. Use the conventions in `docs/design.md` for anything that crosses components, and write units and array shapes (for example `[N, 2]`) in docstrings.
- Every generated sample must be reproducible: record the seed, the configuration, and the solver settings next to the output, as the pore generator's `batch.json` and `sample.json` already do.
- Don't change validation thresholds, train/evaluation splits, baselines, or physical parameters (material properties, load steps) without saying so in the PR description. These are scientific decisions, not refactors.
- Never fabricate results. Don't say a test, simulation, or training run passed unless you ran it and saw the output. Code that runs isn't evidence that the physics is right.

## Data, compute, and confidentiality

- The repository is public. Dr. Chen approved the ET-10 graphite data for it. SiC data from the partner company must never be committed, pasted into issues, or sent to an external AI service.
- Until the sponsors approve specific AI tools and data categories (tracked in `docs/risks.md`), give external AI services only public material, synthetic data, or material explicitly approved for that tool. If you're unsure how a file is classified, stop and ask.
- Never write credentials, tokens, SSH keys, or HPC account details into code, logs, commits, or prompts.
- Don't commit generated output, virtual environments, caches, or large binaries: `results/`, `.venv/`, `__pycache__/`, `*.msh`, `*.sif`, archives. Check `git status` before committing.
- Ask a teammate before submitting HPC jobs (Slurm `sbatch`) or long GPU training runs. They use a shared allocation.

## Workflow

- Follow `CONTRIBUTING.md`: branch off `main`, open a pull request, and get a teammate's review. Never commit or force-push to `main`.
- Keep each PR to one change. Its description says what changed, why, and which checks you ran or couldn't run.
- If your change makes a living doc out of date (a new command, interface, data field, or constraint), update that doc in the same PR.
- `docs/risks.md` is owned by the team. Agents may point out a new or changed risk in the PR description, but must not add, re-rank, or close entries themselves.

## Human review

The teammate who submits AI-generated work must understand it and is accountable for it. A reviewer has to approve every PR before it merges, whoever wrote the code.
