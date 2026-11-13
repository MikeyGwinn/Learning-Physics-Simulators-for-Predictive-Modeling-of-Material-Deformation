# This connects Gunnar's pore generator to the MOOSE input data 

"""
Reads the generator's own outputs, unchanged:
  <sample>/pores.csv               id, x_mm, y_mm, radius_mm, kind (explicit | small)
  <sample>/effective_porosity.csv  x_mm, y_mm, effective_porosity, modulus_factor (one row per 50 um cell)
  <sample>/sample.json             seed, config, statistics
Writes, in the current folder:
  efactor_<seed>.txt   stiffness map E/E0 for MOOSE's PiecewiseMultilinear (um grid):
                       1 in matrix, modulus_factor (= 1 - f_eff) in each 50 um cell, KPORE inside explicit pores
  rve_<seed>.png       picture of the sample (compare paper Fig. 3)
  fig_pore_cdf.png     pore-radius CDF of all samples given vs the paper's log-normal (paper Fig. 2)
Usage: python3 gunnar_to_moose.py <sample_dir> [<sample_dir> ...]
"""
import sys, csv, json, math
from pathlib import Path
import numpy as np

GRID_UM = 2.0      # grid spacing for the stiffness map (finer than the 4 um mesh)
KPORE = 1e-4       # stiffness factor in explicit pores; the paper says only "very low stiffness"

def read(sample):
    s = Path(sample)
    meta = json.loads((s / "sample.json").read_text())
    pores = list(csv.DictReader((s / "pores.csv").open()))
    cells = list(csv.DictReader((s / "effective_porosity.csv").open()))
    return meta, pores, cells

def stiffness_map(meta, pores, cells):
    L = meta["config"]["size_mm"] * 1000.0
    cell = meta["config"].get("effective_cell_um", 50.0)
    n = round(L / cell)
    fac = np.ones((n, n))
    for c in cells:
        i = int(float(c["x_mm"]) * 1000 // cell); j = int(float(c["y_mm"]) * 1000 // cell)
        fac[j, i] = float(c["modulus_factor"])
    xs = np.arange(0, L + GRID_UM / 2, GRID_UM)
    X, Y = np.meshgrid(xs, xs)
    E = fac[np.minimum((Y // cell).astype(int), n - 1), np.minimum((X // cell).astype(int), n - 1)]
    for p in pores:
        if p["kind"] != "explicit": continue
        x, y, r = (float(p[k]) * 1000 for k in ("x_mm", "y_mm", "radius_mm"))
        E[(X - x) ** 2 + (Y - y) ** 2 <= r * r] = KPORE
    return xs, E

if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from math import erf
    samples = sys.argv[1:] or sys.exit(__doc__)
    allr = []
    for sdir in samples:
        meta, pores, cells = read(sdir)
        seed = meta["seed"]; st = meta["statistics"]
        xs, E = stiffness_map(meta, pores, cells)
        with open(f"efactor_{seed}.txt", "w") as fo:
            fo.write("AXIS X\n" + " ".join(f"{v:g}" for v in xs) + "\nAXIS Y\n" + " ".join(f"{v:g}" for v in xs) + "\nDATA\n")
            for row in E: fo.write(" ".join(f"{v:.5g}" for v in row) + "\n")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(E, origin="lower", extent=[0, xs[-1], 0, xs[-1]], cmap="Greys_r", vmin=0, vmax=1)
        ax.set_title(f"Seed {seed}: {st['explicit_pore_count']} explicit pores (porosity {st['explicit_porosity']:.3f})\n"
                     f"+ {st['small_pore_count']} small pores smeared (porosity {st['small_porosity']:.3f})", fontsize=9)
        ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
        fig.tight_layout(); fig.savefig(f"rve_{seed}.png", dpi=160); plt.close(fig)
        allr.append((seed, np.array([float(p["radius_mm"]) * 1000 for p in pores])))
        print(f"seed {seed}: wrote efactor_{seed}.txt and rve_{seed}.png; {st['pore_count']} pores, "
              f"{st['explicit_pore_count']} explicit; porosity {st['achieved_porosity']:.3f}")
    mu, sig = meta["config"]["radius_mu"], meta["config"]["radius_sigma"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    r = np.linspace(0.5, 120, 400)
    ax.plot(r, [0.5 * (1 + erf((math.log(v) - mu) / (math.sqrt(2) * sig))) for v in r], "k-", lw=2,
            label=f"analytical log-normal, mu = {mu}, sigma = {sig} (paper Fig. 2)")
    for k, (seed, rr) in enumerate(allr):
        bins = np.arange(0, 125, 5)
        ax.plot(bins, [np.mean(rr <= b) for b in bins], "os^"[k % 3], ms=5, label=f"seed {seed} ({len(rr)} pores)")
    ax.set_xlabel("pore radius (um)"); ax.set_ylabel("cumulative distribution"); ax.set_xlim(0, 120); ax.set_ylim(0, 1.03)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Pore radii from the team's pore generator vs the paper (5 um bins, as in Fig. 2)", fontsize=10)
    fig.tight_layout(); fig.savefig("fig_pore_cdf.png", dpi=160); plt.close(fig)
