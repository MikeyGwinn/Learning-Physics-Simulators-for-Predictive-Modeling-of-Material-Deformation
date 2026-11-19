""" THIS is the main script to COMPARE the RVE pore graphite test with Chakraborty

Run in the run folder:  python3 viz_sent.py
Makes: fig1_stress_strain.png, fig2_damage_profile.png, fig3_damage_maps.png,
       fig4_crack_growth.png, sent_crack.gif, summary.txt
"""
import glob, os, re, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")

L_ELL = 0.0075  # mm, Table 1
H_ELEM = 0.0035  # mm, element size in sent_paper.i
# Paper Fig. 1b, digitized from the accepted manuscript (strain, stress GPa)
PAPER = [(0.000027,0.0085), (0.000124,0.0191), (0.000221,0.0328), (0.000317,0.0456), (0.000414,0.0593), (0.000511,0.0742), (0.000608,0.0858), (0.000705,0.1007), (0.000802,0.1134), (0.000899,0.1282), (0.000996,0.1409), (0.001092,0.1536), (0.001189,0.1685), (0.001286,0.1812), (0.001383,0.1960), (0.001480,0.2087), (0.001577,0.2236), (0.001674,0.2363), (0.001770,0.2490), (0.001867,0.2628), (0.001964,0.2766), (0.002061,0.2903), (0.002158,0.3041), (0.002255,0.3168), (0.002352,0.3295), (0.002448,0.3433), (0.002545,0.3571), (0.002642,0.3698), (0.002739,0.3836), (0.002836,0.3963), (0.002933,0.4101), (0.003030,0.4228), (0.003126,0.4355), (0.003223,0.4493), (0.003320,0.4630), (0.003417,0.4758), (0.003514,0.4885), (0.003611,0.5012), (0.003708,0.5139), (0.003804,0.5266), (0.003901,0.5393), (0.003998,0.5521), (0.004095,0.5648), (0.004192,0.5775), (0.004289,0.5902), (0.004386,0.6019), (0.004483,0.6146), (0.004579,0.6273), (0.004676,0.6389), (0.004773,0.6495), (0.004870,0.6623), (0.004967,0.6750), (0.005064,0.6877), (0.005161,0.7004), (0.005193,0.7004), (0.005236,0.3507), (0.005268,0.1875), (0.005300,0.0964), (0.005311,0.1007)]

def load_csv(path):
    with open(path) as f:
        r = csv.reader(f); head = next(r)
        rows = [[float(x) for x in row] for row in r if row]
    return head, np.array(rows)

def col(head, arr, name):
    return arr[:, head.index(name)]

main = [p for p in glob.glob("*_out.csv")]
if not main: sys.exit("No *_out.csv here. Run this in the folder with the MOOSE outputs.")
head, A = load_csv(main[0])
strain, stress, maxc = col(head, A, "strain"), col(head, A, "stress"), col(head, A, "max_c")
pe = np.array(PAPER)
ip, ipp = int(np.argmax(stress)), int(np.argmax(pe[:, 1]))
peak, pstrain = stress[ip], strain[ip]
ppeak, ppstrain = pe[ipp, 1], pe[ipp, 0]
# initial slope from the first 30% of the pre-peak curve
m = (strain > 0) & (strain < 0.3 * pstrain)
slope = np.polyfit(strain[m], stress[m], 1)[0] if m.sum() > 2 else float("nan")
mp = (pe[:, 0] > 0) & (pe[:, 0] < 0.3 * ppstrain)
pslope = np.polyfit(pe[mp, 0], pe[mp, 1], 1)[0]

BLUE, ORANGE, GREY = "#1f5f8b", "#d9822b", "#7a7a7a"

# ---- Fig 1: stress-strain vs paper
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(pe[:, 0], pe[:, 1], "o", ms=3.5, color=ORANGE, label="Chakraborty et al. 2016, Fig. 1b (digitized)")
ax.plot(strain, stress, "-", lw=2.2, color=BLUE, label="Our MOOSE run")
ax.annotate(f"paper peak {ppeak:.3f} GPa\nat strain {ppstrain:.5f}", (ppstrain, ppeak), (ppstrain - 0.0027, ppeak - 0.07),
            arrowprops=dict(arrowstyle="->", color=ORANGE), color=ORANGE, fontsize=9)
ax.annotate(f"our peak {peak:.3f} GPa\nat strain {pstrain:.5f}", (pstrain, peak), (pstrain - 0.0027, peak - 0.21),
            arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE, fontsize=9)
err = 100 * (peak - ppeak) / ppeak
broke = stress[-1] < 0.5 * peak   # did the plate actually fail in this run?
ax.set_title(f"SENT benchmark: peak stress differs by {err:+.1f}% (target within 10%)" if broke else f"SENT benchmark: no fracture yet by strain {strain[-1]:.4f} (stress {stress[-1]:.3f} GPa, still rising)", fontsize=11)
ax.set_xlabel("strain (top displacement / specimen height)"); ax.set_ylabel("nominal stress (GPa)")
ax.set_xlim(0, max(0.006, strain.max())); ax.set_ylim(0, max(0.85, 1.1 * stress.max())); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig("fig1_stress_strain.png", dpi=200); plt.close(fig)

# ---- Fig 2: damage profile across the crack vs exp(-|y|/l)
dl = sorted([p for p in glob.glob("*damage_line_*.csv") if re.search(r"_\d+\.csv$", p)], key=lambda p: int(re.findall(r"_(\d+)\.csv$", p)[0]))
leff = float("nan")
if dl:
    h2, D = load_csv(dl[-1]); y, c = col(h2, D, "y"), col(h2, D, "c")
    yy = np.concatenate([-y[::-1], y]); cc = np.concatenate([c[::-1], c])
    fit = (c > 0.05) & (c < 0.9)
    if fit.sum() > 3:
        leff = -1 / np.polyfit(y[fit], np.log(c[fit]), 1)[0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ys = np.linspace(-0.06, 0.06, 600)
    ax.plot(ys, np.exp(-np.abs(ys) / L_ELL), "--", color=ORANGE, lw=2, label=r"Eq. 1: $c = e^{-|y|/l}$, $l$ = 0.0075 mm")
    ax.plot(yy, cc, "-", color=BLUE, lw=2.2, label="Our damage across the crack (x = 0.75 mm, end of run)")
    ax.annotate("", (-L_ELL, 0.37), (L_ELL, 0.37), arrowprops=dict(arrowstyle="<->", color=GREY))
    ax.text(0, 0.40, "2l", ha="center", color=GREY)
    for k in range(1, 4): ax.axvline(k * H_ELEM, color=GREY, lw=.6, alpha=.5); ax.axvline(-k * H_ELEM, color=GREY, lw=.6, alpha=.5)
    ax.text(H_ELEM * 1.1, 0.03, "element edges (h)", fontsize=8, color=GREY)
    ax.set_xlabel("distance across the crack (mm)"); ax.set_ylabel("damage c")
    ax.set_title(f"Damage band width: fitted l = {leff:.4f} mm vs model l = {L_ELL} mm", fontsize=11)
    ax.set_xlim(-0.05, 0.05); ax.set_ylim(0, 1.05); ax.grid(alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig("fig2_damage_profile.png", dpi=200); plt.close(fig)

# ---- snapshots of the full damage field
snaps = sorted([p for p in glob.glob("*field_*.csv") if re.search(r"_\d+\.csv$", p)], key=lambda p: int(re.findall(r"_(\d+)\.csv$", p)[0]))
snaps = [p for p in snaps if sum(1 for _ in open(p)) > 2]   # skip files still being written / empty
def grid(path):
    h3, S = load_csv(path)
    x, y, c = col(h3, S, "x"), col(h3, S, "y"), col(h3, S, "c")
    xs, ys = np.unique(np.round(x, 9)), np.unique(np.round(y, 9))
    G = np.full((len(ys), len(xs)), np.nan)
    G[np.searchsorted(ys, np.round(y, 9)), np.searchsorted(xs, np.round(x, 9))] = c
    full = np.vstack([G[::-1], G[1:]])  # mirror the half model into the full plate
    return xs, np.concatenate([-ys[::-1], ys[1:]]) + ys.max(), full
ZOOM = dict(x=(0.45, 1.0), y=(0.45, 0.55))   # the crack path; the damage band is only ~0.02 mm wide
def draw(a, xs, ys, G):
    im = a.pcolormesh(xs, ys, G, shading="auto", cmap="turbo", vmin=0, vmax=1)
    a.plot([0, 0.5], [0.5, 0.5], color="w", lw=1.2)
    a.set_xlim(*ZOOM["x"]); a.set_ylim(*ZOOM["y"]); a.set_aspect("equal"); a.set_xticks([0.5, 0.75, 1.0]); a.set_yticks([])
    return im
def crack_tip(xs, ys, G):
    mid = np.argmin(np.abs(ys - ys.max() / 2)); row = G[mid]
    cracked = xs[row > 0.95]
    return cracked.max() if cracked.size else 0.5
# map each snapshot file to its time: use the *_time.csv index MOOSE writes (time_data = true) if present
tfile = [p for p in glob.glob("*field_time.csv")]
tmap = {}
if tfile:
    ht, T = load_csv(tfile[0])
    tcol = T[:, ht.index("time")]; scol = T[:, ht.index("timestep")]
    tmap = {int(sn): tt for sn, tt in zip(scol, tcol)}
times = col(head, A, "time")
info = []
for p in snaps:
    num = int(re.findall(r"_(\d+)\.csv$", p)[0])
    t = tmap.get(num, times[num] if num < len(times) else None)
    if t is None: continue
    e, s = np.interp(t, times, strain), np.interp(t, times, stress)
    xs, ys, G = grid(p); info.append((num, e, s, crack_tip(xs, ys, G), p))
if info:
    pick = []
    targets = [0.6 * pstrain, pstrain]
    for t in targets: pick.append(min(info, key=lambda r: abs(r[1] - t)))
    after = [r for r in info if r[1] > pstrain]
    if after:
        mid = min(after, key=lambda r: abs(r[3] - 0.75)); pick.append(mid); pick.append(after[-1])
    seen = []; pick = [r for r in pick if not (r[0] in seen or seen.append(r[0]))]
    fig, axs = plt.subplots(len(pick), 1, figsize=(7.5, 1.55 * len(pick) + 0.8))
    axs = np.atleast_1d(axs)
    for a, (step, e, s, tip, p) in zip(axs, pick):
        im = draw(a, *grid(p))
        a.set_title(f"strain {e:.5f}, stress {s:.3f} GPa, crack tip x = {tip:.2f} mm", fontsize=9)
    axs[-1].set_xlabel("x (mm); band shown is y = 0.45–0.55 mm around the notch line")
    fig.colorbar(im, ax=axs.tolist(), shrink=.8, label="damage c")
    fig.suptitle("Crack growth along the notch line (compare paper Fig. 1c); white = initial notch", fontsize=11)
    fig.savefig("fig3_damage_maps.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- Fig 4: crack tip vs strain, with the stress curve
    fig, ax = plt.subplots(figsize=(7.5, 4.3)); ax2 = ax.twinx()
    ax.plot(strain, stress, color=BLUE, lw=2); ax.set_ylabel("stress (GPa)", color=BLUE)
    ax2.plot([r[1] for r in info], [r[3] for r in info], "s-", color=ORANGE, ms=4); ax2.set_ylabel("crack tip x (mm)", color=ORANGE)
    ax2.set_ylim(0.45, 1.02); ax.set_xlabel("strain"); ax.grid(alpha=.3)
    ax.set_title("The crack waits at the notch, then runs across in a few steps (unstable growth)" if broke else "Crack tip vs load: the crack has not left the notch in this run", fontsize=11)
    fig.tight_layout(); fig.savefig("fig4_crack_growth.png", dpi=200); plt.close(fig)

    # ---- animation
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 6.5), gridspec_kw=dict(height_ratios=[1, 2.2]))
        draw(a, *grid(info[0][4])); a.set_title("damage c along the notch line")
        b.plot(pe[:, 0], pe[:, 1], "o", ms=2.5, color=ORANGE, label="paper"); b.plot(strain, stress, color=BLUE, lw=1.5, label="ours")
        dot, = b.plot([], [], "o", color="k"); b.set_xlabel("strain"); b.set_ylabel("stress (GPa)"); b.legend(fontsize=8); b.grid(alpha=.3)
        def upd(k):
            step, e, s, tip, p = info[k]; a.clear(); draw(a, *grid(p)); a.set_title(f"damage c, strain {e:.5f}", fontsize=9)
            dot.set_data([e], [s]); return []
        FuncAnimation(fig, upd, frames=len(info), interval=150).save("sent_crack.gif", writer=PillowWriter(fps=8))
        plt.close(fig)
    except Exception as ex:
        print("animation skipped:", ex)

with open("summary.txt", "w") as f:
    rows = [("Peak stress (GPa)", ppeak, peak), ("Strain at peak", ppstrain, pstrain),
            ("Initial slope (GPa)", pslope, slope), ("Damage band length l (mm)", L_ELL, leff)]
    f.write(f"{'quantity':28s}{'paper':>12s}{'ours':>12s}{'diff %':>10s}\n")
    for n, a, b in rows: f.write(f"{n:28s}{a:12.5g}{b:12.5g}{100*(b-a)/a:10.1f}\n")
    f.write(f"\nsteps: {len(strain)-1}, final strain {strain[-1]:.5f}, final max_c {maxc[-1]:.3f}\n")
print(open("summary.txt").read())
