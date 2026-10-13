# SEM Mesh Studio

Create a finite-element mesh from Skärberg SEM images, run horizontal-tension
phase-field fracture on that exact mesh, and view the resulting animation.

```bash
cd /nfs/hpc/share/scumnioj/PhaseFieldTest
source activate.sh
python sem_meshing_gui.py --port 8765
```

Open <http://127.0.0.1:8765>. The launcher replaces an existing process on the
selected port. For remote access, forward that port from your workstation:

```bash
ssh -L 8765:127.0.0.1:8765 YOUR_USER@THE_NODE_RUNNING_THE_GUI
```

The GUI uses raw intensity threshold segmentation, a colored ROI histogram,
and WebGL2 mesh rendering. Build a mesh connected between both grips, then
press **Simulate fracture**. Accepted frames are animated even if convergence
limits stop the solve, with partial results labeled. 

See [SEM_GUI.md](SEM_GUI.md) for controls, scientific assumptions, dataset
provenance, artifacts and the full image-to-fracture benchmark.

The benchmark defaults to planning only; execution requires `--execute`:

```bash
python image_workflow_benchmark.py --dry-run
```

Runtime scripts: `sem_meshing_gui.py`, `sem_geometry.py`, `fetch_skarberg.py`,
`simulate_fracture.py`, `bounded_damage.py`, `scientific_common.py`,
`scientific_jit.py`, `visualize.py`, and `activate.sh`.
SEM solves require the generated image mesh. Benchmark scripts are
`image_workflow_benchmark.py`, `image_workflow_worker.py`,
`plot_image_workflow.py`, and `benchmark.py` (shared agreement validation).

The environment `.conda`, upstream `phasefieldx` checkout, public dataset and
simulation results are retained. Standalone scripts unrelated to the SEM
workflow and their documentation/configs were moved to the recoverable sibling
directory `/nfs/hpc/share/scumnioj/PhaseFieldTest_retired_scripts_20261013/`.

Verification-only scripts under `tests/` and the synthetic-mesh generator were
deleted during cleanup. They are not required for the SEM workflow.
