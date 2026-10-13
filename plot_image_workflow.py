"""Plot completed image-workflow measurements; never starts a benchmark."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def plot(folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(folder)
    with (folder/"measurements.csv").open() as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if r["status"] == "complete" and int(r["repeat"]) >= 0]
    if not valid:
        raise ValueError("No converged measured runs; warmups and failures are excluded")
    output = folder/"plots"
    output.mkdir(exist_ok=True)
    stage_keys = ["image_segmentation_s", "contour_s", "mesh_s", "fracture_setup_s", "fracture_compute_s"]
    for image in sorted({r["image_id"] for r in valid}):
        selected = [r for r in valid if r["image_id"] == image]
        groups = defaultdict(list)
        for r in selected:
            groups[(int(r["total_dofs"]), r["backend"])].append(r)
        stats = {}
        fig, (time_ax, speed_ax) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for backend in ("cpu", "gpu"):
            ns = sorted(n for n, b in groups if b == backend)
            if not ns:
                continue
            medians, lows, highs = [], [], []
            for n in ns:
                values = [float(r["process_wall_seconds"]) for r in groups[n, backend]]
                q = np.percentile(values, [25, 50, 75])
                lows.append(q[1]-q[0]); medians.append(q[1]); highs.append(q[2]-q[1])
                stats[n, backend] = q[1]
            time_ax.errorbar(ns, medians, yerr=[lows, highs], marker="o", capsize=4, label=backend.upper())
        ns = sorted(n for n, b in stats if b == "cpu" and (n, "gpu") in stats)
        if ns:
            speed_ax.plot(ns, [stats[n, "cpu"]/stats[n, "gpu"] for n in ns], "o-")
        speed_ax.axhline(1, color="gray", linestyle="--")
        time_ax.set(xscale="log", yscale="log", xlabel="Total DOFs", ylabel="Image-to-fracture process wall time [s]")
        speed_ax.set(xscale="log", xlabel="Total DOFs", ylabel="CPU median / GPU median")
        time_ax.legend(title="Median ± interquartile range")
        for ax in (time_ax, speed_ax):
            ax.grid(alpha=.2, which="both")
        failures = sum(r["status"] != "complete" and int(r["repeat"]) >= 0 for r in rows if r["image_id"] == image)
        fig.suptitle(f"{image}: full image → mesh → fracture workflow · {failures} failed measurements excluded")
        fig.savefig(output/f"{image}_timing.png", dpi=180)
        fig.savefig(output/f"{image}_timing.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
        for ax, backend in zip(axes, ("cpu", "gpu")):
            ns = sorted(n for n, b in groups if b == backend)
            bottom = np.zeros(len(ns))
            for key, label in list(zip(stage_keys, ["Image / mask", "Contours", "Meshing", "FEM setup", "Fracture solve"]))+[("other", "Startup / output / other")]:
                values = []
                for n in ns:
                    samples = groups[n, backend]
                    values.append(np.mean([float(r[key]) if key != "other" else
                                  max(0, float(r["process_wall_seconds"])-sum(float(r[k]) for k in stage_keys)) for r in samples]))
                ax.bar([str(n) for n in ns], values, bottom=bottom, label=label)
                bottom += values
            ax.set(title=backend.upper(), xlabel="Total DOFs", ylabel="Mean stage time [s]")
            ax.tick_params(axis="x", rotation=25)
        axes[-1].legend(fontsize=8)
        fig.suptitle(f"{image}: stage costs, including process startup and output")
        fig.savefig(output/f"{image}_stages.png", dpi=180)
        fig.savefig(output/f"{image}_stages.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    plot(parser.parse_args().folder)
