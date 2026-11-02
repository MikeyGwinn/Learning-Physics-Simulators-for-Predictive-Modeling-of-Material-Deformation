# Distributed image conversion benchmark

The benchmark queues 300 Skärberg SEM TIFFs: indices 0, 2, ..., 198 from each
of HPC22, HPC30, and HPC45. It uses the raw images and the `SEMImage` threshold
pipeline; author CNN masks and physics solvers are not used. One image produces
one mesh. Selected TIFFs are cached and verified against the published ZIP's
member CRC and a SHA256 recorded in the run manifest. Downloads finish before
any jobs are submitted and are excluded from timing.

Run from the SEM2msh checkout with a shared Python environment containing the
package dependencies and Gmsh. On this cluster, the existing
`PhaseFieldTest/.conda/bin/python` has those dependencies. Choose the image
area explicitly:

```bash
/nfs/hpc/share/scumnioj/PhaseFieldTest/.conda/bin/python -m benchmark.run --crop 1400 900 384 384 --dry-run
/nfs/hpc/share/scumnioj/PhaseFieldTest/.conda/bin/python -m benchmark.run --crop 1400 900 384 384
```

`python benchmark/run.py --crop 1400 900 384 384` also works from the
checkout. If the shell's Python is too old or lacks the meshing dependencies,
the script switches to the sibling `PhaseFieldTest/.conda` Python when it is
available. Running without `--crop` or `--full-image` displays the required
area choice and does not start the benchmark.

Use `--full-image` in place of `--crop LEFT TOP WIDTH HEIGHT` to process whole
TIFFs. That can create far larger contours and may fail the API's contour
limit; such failures are reported per image. The default command stages the
TIFFs, creates a run under `benchmark/results/`, and submits a Slurm array to
the `share` partition. The array starts four workers on four exclusive nodes.
Each worker claims images from a shared queue and keeps processing until the
queue is empty, so Python and Slurm startup happen once per worker. A dependent aggregation
job writes `measurements.csv` and `summary.json` after all array tasks finish.
The run manifest records the submitted job IDs and queue path. Slurm logs are
in `logs/`. A resumed run creates a new queue containing only incomplete images.

For a run-level performance baseline, add `--wait-for-all-nodes`. This requests
all four exclusive nodes in one Slurm job and starts one worker per node after
the full allocation is ready. The resulting `summary.json` records elapsed
conversion wall time and attempted images per second. This time excludes the
Slurm queue wait and image downloads. A resumed run keeps its previous allocation
mode unless `--wait-for-all-nodes` or `--no-wait-for-all-nodes` is supplied.

Use `--partitions` to select from `share`, `dgxh`, `dgx2`, `eecs`, `gpu`, and
`ampere`, for example `--partitions "share,dgxh"`. Slurm expects a comma-separated
argument; shell quotes surround the whole argument and are not passed to Slurm.
The default is `share,dgx2`, the two-partition combination Slurm accepted for
the four-node CPU baseline. On this cluster, `eecs` and `gpu` cannot supply four
nodes in one allocation; `ampere` requires GPU resources, and combining `dgxh`
with the default list produced a scheduler estimate much later than `share,dgx2`.
The benchmark does not request GPUs for its CPU meshing work.

The default settings mirror the GUI's threshold path: 10 nm/pixel, threshold
0.45 with dark pixels void, solid speck removal 16 pixels, filling pores up to
8 pixels, all solid components, 0.5 pixel contour tolerance, 30 nm boundary
mesh size, 80 nm interior size, 100 nm refinement distance, frontal Delaunay,
and five smoothing passes. The crop is never defaulted. An optional JSON file
can override the other settings, for example:

```json
{
  "segmentation": {"threshold": 0.4, "component": "largest"},
  "meshing": {"interior_size_nm": 100, "boundary_size_nm": 40}
}
```

Pass it with `--config settings.json`. Cache and result roots can be changed
with `--cache` and `--output-root`; `--archive` accepts a local copy of the
published ZIP. The script can also seed its cache from the sibling GUI dataset
read-only, when present; use `--seed-dir` to select another raw TIFF directory.

To choose a separate intensity cutoff for each image, pass
`--threshold-percentile 30` or set `segmentation.threshold_percentile` in the
configuration file. The percentile is calculated inside each crop; a value of
30 selects approximately the brightest 70% of pixels as solid before cleanup.
The per-image report records the actual normalized cutoff in `metadata.threshold`
and the raw TIFF cutoff in `metadata.threshold_raw`. This changes the meaning of
the benchmark from a fixed brightness threshold to an assumed solid fraction.
Inspect representative masks before using the geometry quantitatively.

Each `tasks/IMAGE_ID/` contains `mesh.msh`, a 2048-pixel-longest-side PNG of
mesh edges on white, and `report.json`. Reports include stage times, total
conversion time, host, Slurm IDs, source hash, mesh counts, quality, and any
failure. Total per-image time starts at worker configuration and ends after PNG
writing; it excludes queue wait, downloads, and Python startup. Run-level wall
time spans the first image start through the last report. The output PNG is a visual
check, not a complete geometric validation.

After the previous array and aggregation job have finished, requeue only
failed or missing conversions with:

```bash
python -m benchmark.run --resume benchmark/results/RUN_ID --dry-run
python -m benchmark.run --resume benchmark/results/RUN_ID
```

The resume command preserves the original configuration and source hashes.
It refuses to overlap an active prior Slurm submission. The follow-up
aggregation job refreshes the CSV and summary.

If image staging completed but `sbatch` rejected the initial submission, use
`--resume` with that run directory. It submits the saved task manifest without
fetching or modifying the image cache.
