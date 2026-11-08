#Digitizes the results of Chakraborty's papers

"""Compare a MOOSE RVE run with Chakraborty, Sabharwall & Carroll (2016), Figs. 3-5.

Run in the RVE folder:  python3 viz_rve.py rve_h8 1      (run name = input file name without .i, seed)
Makes <run>_fig1_stress_strain.png, <run>_fig2_damage_maps.png, <run>_crack.gif, <run>_summary.txt
"""
import glob, re, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

run = sys.argv[1] if len(sys.argv) > 1 else "rve_paper"
seed = sys.argv[2] if len(sys.argv) > 2 else "1"
# Paper Fig. 4, digitized from the accepted manuscript (strain, stress MPa); circles on the curve add small artifacts
PAPER4 = [(3.01724e-05,0.1203), (6.46552e-05,0.2407), (9.91379e-05,0.3954), (0.000133621,0.5501), (0.000168103,0.7049), (0.000202586,0.8596), (0.000237069,1.014), (0.000271552,1.152), (0.000306034,1.289), (0.000340517,1.427), (0.000375,1.599), (0.000409483,1.736), (0.000443966,1.891), (0.000478448,2.046), (0.000512931,2.183), (0.000547414,2.338), (0.000581897,2.476), (0.000616379,2.63), (0.000650862,2.768), (0.000685345,2.94), (0.000719828,3.06), (0.00075431,3.215), (0.000788793,3.352), (0.000823276,3.507), (0.000857759,3.662), (0.000892241,3.782), (0.000926724,3.937), (0.000961207,4.074), (0.00099569,4.246), (0.00101293,4.298), (0.00104741,4.453), (0.0010819,4.573), (0.00111638,4.728), (0.00115086,4.865), (0.00118534,5.003), (0.00121983,5.14), (0.00125431,5.295), (0.00128879,5.415), (0.00132328,5.57), (0.00135776,5.708), (0.00139224,5.845), (0.00142672,5.983), (0.00146121,6.103), (0.00149569,6.275), (0.00153017,6.378), (0.00156466,6.516), (0.00159914,6.653), (0.00163362,6.774), (0.0016681,6.928), (0.00170259,7.049), (0.00173707,7.186), (0.00177155,7.307), (0.00180603,7.444), (0.00184052,7.582), (0.001875,7.719), (0.00190948,7.822), (0.00194397,7.943), (0.00197845,8.063), (0.00200431,8.183), (0.00203017,8.269), (0.00206466,8.407), (0.00209914,8.51), (0.00213362,8.63), (0.0021681,8.751), (0.00220259,8.871), (0.00223707,8.991), (0.00227155,9.112), (0.00235776,9.404), (0.00239224,9.49), (0.00242672,9.61), (0.00246121,9.713), (0.00249569,9.851), (0.00253017,9.92), (0.00256466,10.06), (0.00259914,10.14), (0.00260776,10.16), (0.00261638,10.19), (0.002625,10.23), (0.00263362,10.25), (0.00264224,10.28), (0.00265086,10.3), (0.00265948,10.33), (0.0026681,10.35), (0.00267672,10.37), (0.00268534,10.4), (0.00269397,10.42), (0.00270259,10.45), (0.00271121,10.47), (0.00271983,10.49), (0.00272845,10.5), (0.00273707,10.54), (0.00274569,10.57), (0.00275431,10.59), (0.00276293,10.59), (0.00277155,10.57), (0.00278017,10.52), (0.00278879,10.64), (0.00278879,10.4), (0.00279741,10.61), (0.00279741,10.38), (0.00280603,10.61), (0.00280603,10.38), (0.00281466,10.68), (0.00281466,10.57), (0.00281466,10.33), (0.00282328,10.68), (0.00282328,10.54), (0.00282328,10.3), (0.0028319,10.66), (0.0028319,10.35), (0.00284052,10.66), (0.00284052,10.37), (0.00284914,10.61), (0.00284914,10.4), (0.00285776,10.61), (0.00285776,10.33), (0.00286638,10.4), (0.002875,10.42), (0.00288362,10.44), (0.00289224,10.47), (0.00290086,10.56), (0.00290948,10.64), (0.00290948,10.47), (0.0029181,10.69), (0.0029181,10.47), (0.00292672,10.69), (0.00292672,10.47), (0.00293534,10.71), (0.00293534,10.54), (0.00293534,10.4), (0.00294397,10.71), (0.00294397,10.57), (0.00294397,10.4), (0.00295259,10.64), (0.00295259,10.42), (0.00296121,10.64), (0.00296121,10.42), (0.00296983,10.64), (0.00296983,10.47), (0.00296983,10.04), (0.00297845,10.56), (0.00297845,10.13), (0.00297845,9.954), (0.00298707,10.62), (0.00298707,10.18), (0.00298707,9.903), (0.00299569,10.62), (0.00299569,10.18), (0.00299569,9.903), (0.00299569,7.427), (0.00300431,10.64), (0.00300431,10.02), (0.00300431,9.885), (0.00300431,7.513), (0.00300431,7.341), (0.00301293,10.61), (0.00301293,9.576), (0.00301293,7.564), (0.00301293,7.289), (0.00302155,10.18), (0.00302155,10.02), (0.00302155,9.163), (0.00302155,7.564), (0.00302155,7.289), (0.00303017,10.18), (0.00303017,9.903), (0.00303017,9.576), (0.00303017,7.444), (0.00303017,7.272), (0.00303879,10.13), (0.00303879,9.954), (0.00303879,9.129), (0.00303879,6.585), (0.00304741,10.04), (0.00304741,7.564), (0.00304741,7.41), (0.00304741,6.172), (0.00305603,7.564), (0.00305603,7.289), (0.00305603,6.516), (0.00305603,6.172), (0.00305603,5.862), (0.00305603,5.484), (0.00306466,7.513), (0.00306466,7.341), (0.00306466,5.794), (0.00306466,5.175), (0.00307328,7.427), (0.00307328,5.415), (0.00307328,5.003), (0.0030819,5.175), (0.0030819,4.968), (0.00309052,5.003), (0.00309052,4.797), (0.00309914,4.968), (0.00309914,4.659), (0.00310776,4.762), (0.00310776,4.418), (0.00311638,4.59), (0.00311638,4.35), (0.003125,4.384), (0.003125,3.868), (0.00313362,4.315), (0.00313362,2.699), (0.00314224,3.834), (0.00314224,2.527), (0.00315086,2.63), (0.00315086,2.355), (0.00315948,2.407), (0.0031681,2.338), (0.00317672,2.338), (0.00318534,2.338), (0.00319397,2.338), (0.00320259,2.338), (0.00321121,2.338), (0.00321983,2.338), (0.00322845,2.338), (0.00323707,2.338), (0.00324569,2.338), (0.00325431,2.338), (0.00326293,2.338), (0.00327155,2.338), (0.00328017,2.338), (0.00328879,2.338), (0.00329741,2.338), (0.00330603,2.338), (0.00331466,2.338), (0.00332328,2.338), (0.0033319,2.338), (0.00334052,2.338), (0.00334914,2.338), (0.00335776,2.338), (0.00336638,2.338), (0.003375,2.355), (0.00338362,2.372), (0.00339224,2.372), (0.00340086,2.372), (0.00340948,2.372), (0.0034181,2.372), (0.00342672,2.355), (0.00343534,2.338), (0.00344397,2.304), (0.00345259,2.321), (0.00345259,2.115), (0.00346121,2.252), (0.00346121,1.221), (0.00346983,2.08), (0.00346983,1.186), (0.00347845,1.221)]

def load(path):
    with open(path) as f:
        r = csv.reader(f); h = next(r); return h, np.array([[float(x) for x in row] for row in r if row])
h, A = load(f"{run}_out.csv")
c = lambda name: A[:, h.index(name)]
t, strain, stress = c("time"), c("strain"), c("stress")
pe = np.array(PAPER4)
ip, jp = int(np.argmax(stress)), int(np.argmax(pe[:, 1]))
m = (strain > 0) & (strain < 0.4 * strain[ip]); slope = np.polyfit(strain[m], stress[m], 1)[0] if m.sum() > 2 else np.nan
mp = (pe[:, 0] > 0) & (pe[:, 0] < 0.4 * pe[jp, 0]); pslope = np.polyfit(pe[mp, 0], pe[mp, 1], 1)[0]
BLUE, ORANGE = "#1f5f8b", "#d9822b"

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(pe[:, 0], pe[:, 1], "o", ms=3, color=ORANGE, label="Chakraborty et al. 2016, Fig. 4 (their RVE, digitized)")
ax.plot(strain, stress, "-", lw=2.2, color=BLUE, label=f"Our MOOSE run ({run}, pore seed {seed})")
err = 100 * (stress[ip] - pe[jp, 1]) / pe[jp, 1]
ax.set_title(f"Graphite RVE: our peak {stress[ip]:.2f} MPa vs paper {pe[jp,1]:.2f} MPa ({err:+.0f}%)", fontsize=11)
ax.set_xlabel("strain"); ax.set_ylabel("stress (MPa)"); ax.set_xlim(0, max(0.004, strain.max())); ax.set_ylim(0, 13)
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.text(0.30, 0.04, "Different random pore layouts, so curves should agree\nin stiffness and peak range, not in every wiggle.",
        transform=ax.transAxes, ha="left", fontsize=8, color="#555")
fig.tight_layout(); fig.savefig(f"{run}_fig1_stress_strain.png", dpi=200); plt.close(fig)

pores = []
try:
    for p in csv.DictReader(open(f"samples/seed-{seed}/pores.csv")):
        if p["kind"] == "explicit": pores.append((float(p["x_mm"]) * 1e3, float(p["y_mm"]) * 1e3, float(p["radius_mm"]) * 1e3))
except FileNotFoundError:
    pass
snaps = sorted([p for p in glob.glob(f"{run}_*field_*.csv") if re.fullmatch(re.escape(run) + r"_(?:out_)?(?:snap_)?field_\d+\.csv", p)], key=lambda p: int(re.findall(r"_(\d+)\.csv$", p)[0]))   # this run only
snaps = [p for p in snaps if sum(1 for _ in open(p)) > 2]   # skip files still being written / empty
tf = [p for p in glob.glob(f"{run}_*field_time.csv") if re.fullmatch(re.escape(run) + r"_(?:out_)?(?:snap_)?field_time\.csv", p)]; tmap = {}
if tf:
    ht, T = load(tf[0]); tmap = {int(s): v for s, v in zip(T[:, ht.index("timestep")], T[:, ht.index("time")])}
def grid(path):
    hh, S = load(path); x, y, cc = (S[:, hh.index(k)] for k in ("x", "y", "c"))
    xs, ys = np.unique(np.round(x, 6)), np.unique(np.round(y, 6)); G = np.full((len(ys), len(xs)), np.nan)
    G[np.searchsorted(ys, np.round(y, 6)), np.searchsorted(xs, np.round(x, 6))] = cc; return xs, ys, G
info = []
for p in snaps:
    n = int(re.findall(r"_(\d+)\.csv$", p)[0]); tt = tmap.get(n, t[n] if n < len(t) else None)
    if tt is None: continue
    info.append((n, np.interp(tt, t, strain), np.interp(tt, t, stress), p))
def draw(a, p, e, s):
    xs, ys, G = grid(p)
    im = a.imshow(G, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="jet", vmin=0, vmax=1)
    for x, y, r in pores: a.add_patch(Circle((x, y), r, facecolor="white", edgecolor="none"))
    a.set_title(f"strain {e:.4f}, {s:.2f} MPa", fontsize=9); a.set_xticks([]); a.set_yticks([]); return im
if info:
    # like the paper's Fig. 5: before the peak, at the peak, during unloading, final
    after = [r for r in info if r[1] > strain[ip]]
    pick = [min(info, key=lambda r: abs(r[1] - 0.7 * strain[ip])), min(info, key=lambda r: abs(r[1] - strain[ip]))]
    if after: pick += [after[len(after) // 2], after[-1]]
    fig, axs = plt.subplots(1, len(pick), figsize=(3.4 * len(pick), 3.7)); axs = np.atleast_1d(axs)
    for a, (n, e, s, p) in zip(axs, pick): im = draw(a, p, e, s)
    fig.colorbar(im, ax=axs.tolist(), shrink=.8, label="damage c")
    fig.suptitle("Damage in our RVE (compare paper Fig. 5): cracks start at pores and link up", fontsize=11)
    fig.savefig(f"{run}_fig2_damage_maps.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4.2))
        b.plot(pe[:, 0], pe[:, 1], "o", ms=2, color=ORANGE, label="paper"); b.plot(strain, stress, color=BLUE, label="ours")
        dot, = b.plot([], [], "ko"); b.set_xlabel("strain"); b.set_ylabel("stress (MPa)"); b.grid(alpha=.3); b.legend(fontsize=8)
        def upd(k):
            a.clear(); n, e, s, p = info[k]; draw(a, p, e, s); dot.set_data([e], [s]); return []
        FuncAnimation(fig, upd, frames=len(info), interval=150).save(f"{run}_crack.gif", writer=PillowWriter(fps=6)); plt.close(fig)
    except Exception as ex:
        print("animation skipped:", ex)
with open(f"{run}_summary.txt", "w") as f:
    f.write(f"{'quantity':30s}{'paper':>10s}{'ours':>10s}{'diff %':>9s}\n")
    for n_, a_, b_ in [("Peak stress (MPa)", pe[jp, 1], stress[ip]), ("Strain at peak", pe[jp, 0], strain[ip]), ("Initial slope (MPa)", pslope, slope)]:
        f.write(f"{n_:30s}{a_:10.4g}{b_:10.4g}{100*(b_-a_)/a_:9.1f}\n")
    f.write(f"\nrun {run}, pore seed {seed}, {len(t)-1} steps, final strain {strain[-1]:.4f}\n")
print(f"{len(snaps)} damage snapshots found, {len(info)} used for maps/GIF")
print(open(f"{run}_summary.txt").read())
