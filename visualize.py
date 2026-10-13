"""Render fracture snapshots, force/energy plots and a synchronized MP4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def render(run, output=None, scale=5.0, fps=12, movie=True, allow_partial=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.tri import Triangulation

    run = Path(run)
    output = Path(output) if output else run / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads((run / "status.json").read_text())
    partial = status["status"] != "complete"
    if partial and (not allow_partial or status["status"] != "failed"):
        raise ValueError("Incomplete simulations require --allow-partial and saved accepted steps")
    frames = json.loads((run / "frames.json").read_text())
    if not frames:
        raise ValueError("No fields saved; rerun without --no-fields")
    with (run / "history.csv").open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No accepted load steps available to animate")
    accepted = {0, *(int(row["step"]) for row in rows)}
    if any(frame["step"] not in accepted or not (run/"fields"/frame["file"]).is_file() for frame in frames):
        raise ValueError("Saved frames must correspond to accepted load steps")
    history = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    mesh = np.load(run / "fields/mesh.npz")
    coords, triangles = mesh["points"][:, :2], mesh["triangles"]
    geo = json.loads((run / "geometry.json").read_text())
    by_step = {int(row["step"]): i for i, row in enumerate(rows)}
    sep = np.r_[0, history["separation"]]
    force = np.r_[0, history["force"]]
    fig, (ax, curve) = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    title = "Porous material under horizontal tension"
    if partial:
        import textwrap
        title += " — PARTIAL RESULT\n" + textwrap.fill(status.get("reason", "Solver stopped early"), 110)
    fig.suptitle(title, fontsize=10 if partial else 12)
    curve.plot(sep, force, color="0.65", lw=1.5, label="Accepted load path" if partial else "Complete load path")
    progress, = curve.plot([], [], color="tab:blue", lw=2)
    cursor, = curve.plot([], [], "o", color="tab:red")
    curve.set(xlabel="Grip separation [mm]", ylabel="Reaction force [N]")
    curve.grid(alpha=0.25)
    curve.legend(loc="best")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="inferno"), ax=ax, label="Damage d (0 intact, 1 fractured)", shrink=0.8)
    pad = max(geo["width"], geo["height"])*0.08+scale*max(abs(sep))/2

    def draw(frame):
        fields = np.load(run / "fields" / frame["file"])
        displaced = coords+scale*fields["displacement"]
        tri = Triangulation(displaced[:, 0], displaced[:, 1], triangles)
        ax.clear()
        ax.tripcolor(tri, fields["damage"], shading="gouraud", cmap="inferno", vmin=0, vmax=1, rasterized=True)
        if fields["damage"].min() < 0.9 < fields["damage"].max():
            ax.tricontour(tri, fields["damage"], levels=[0.9], colors="cyan", linewidths=0.6)
        ax.set(xlim=(-pad, geo["width"]+pad), ylim=(-pad, geo["height"]+pad),
               xlabel="x [mm]", ylabel="y [mm]", aspect="equal",
               title=f"Separation {frame['separation']:.5f} mm; deformation ×{scale:g}")
        mid = geo["height"]/2
        for edge, sign in [(0, -1), (geo["width"], 1)]:
            ax.annotate("", xy=(edge+sign*pad*0.8, mid), xytext=(edge+sign*pad*0.2, mid),
                        arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 2})
        ax.text(0.5, 0.97, "Top and bottom traction-free", transform=ax.transAxes, ha="center", va="top", fontsize=9)
        i = 0 if frame["step"] == 0 else by_step[frame["step"]]+1
        progress.set_data(sep[:i+1], force[:i+1])
        cursor.set_data([sep[i]], [force[i]])

    damage_max = [float(np.load(run / "fields" / f["file"])["damage"].max()) for f in frames]
    initiation = next((i for i, v in enumerate(damage_max) if v >= 0.5), len(frames)//2)
    selected = {"initial": 0, "initiation": initiation, "propagation": min(initiation+max(1, (len(frames)-initiation)//2), len(frames)-1), "final": len(frames)-1}
    for name, idx in selected.items():
        draw(frames[idx])
        fig.savefig(output / f"{name}.png", dpi=180)
    if movie:
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg is required for MP4; use --no-movie for PNG/PDF output")
        # Slow short runs uniformly rather than ending the movie abruptly.
        # Preserve every accepted state; no additional solution states are made.
        playback_fps = min(fps, len(frames)/5.0)
        writer = FFMpegWriter(fps=playback_fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        with writer.saving(fig, str(output / "fracture.mp4"), dpi=120):
            for frame in frames:
                draw(frame)
                writer.grab_frame()
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    axes[0].plot(sep, force)
    axes[0].set(xlabel="Grip separation [mm]", ylabel="Force [N]")
    for label, key in [("Elastic", "elastic_energy"), ("Fracture", "fracture_energy"), ("External work", "external_work")]:
        axes[1].plot(history["separation"], history[key], label=label)
    axes[1].set(xlabel="Grip separation [mm]", ylabel="Energy [N mm]")
    axes[1].legend()
    axes[2].plot(history["separation"], history["energy_balance_error"])
    axes[2].set(xlabel="Grip separation [mm]", ylabel="Work − stored/fracture energy [N mm]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.savefig(output / "response.png", dpi=180)
    fig.savefig(output / "response.pdf")
    plt.close(fig)
    print(f"Visualization written to {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--deformation-scale", type=float, default=5)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--no-movie", action="store_true")
    p.add_argument("--allow-partial", action="store_true", help="Animate accepted steps from a failed run; label output as partial")
    a = p.parse_args()
    if not np.isfinite(a.deformation_scale) or a.deformation_scale < 0 or a.fps < 1:
        p.error("Deformation scale must be finite/nonnegative and fps positive")
    render(a.run, a.output, a.deformation_scale, a.fps, not a.no_movie, a.allow_partial)


if __name__ == "__main__":
    main()
