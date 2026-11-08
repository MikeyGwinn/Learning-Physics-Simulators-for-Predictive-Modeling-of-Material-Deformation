"""Mesh study for the SENT benchmark (Check 2): compare runs with different element sizes.

Run in sent_paper/:  python3 compare_sent.py h_l2 h_l4 h_l8     (each folder has its own sent_paper.i and outputs)
Makes fig5_mesh_study.png and mesh_study.txt
"""
import sys, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ast
_src = open("viz_sent.py").read()   # reuse the digitized paper curve without running that script
PAPER = ast.literal_eval(re.search(r"^PAPER = (\[.*\])$", _src, re.M).group(1))
L_ELL = 0.0075

runs = []
for d in sys.argv[1:]:
    h = float(re.search(r"^h\s*=\s*([0-9.eE+-]+)", open(f"{d}/sent_paper.i").read(), re.M).group(1))
    with open(f"{d}/sent_paper_out.csv") as f:
        r = csv.reader(f); head = next(r); A = np.array([[float(x) for x in row] for row in r if row])
    e, s = A[:, head.index("strain")], A[:, head.index("stress")]
    runs.append((d, h, e, s, s.max(), e[s.argmax()], s[-1] < 0.5 * s.max()))
runs.sort(key=lambda r: -r[1])
pe = np.array(PAPER); ppeak = pe[:, 1].max()
fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6))
a.plot(pe[:, 0], pe[:, 1], "o", ms=3, color="#d9822b", label="paper Fig. 1b")
cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
for (d, h, e, s, pk, ep, broke), c in zip(runs, cols):
    a.plot(e, s, color=c, lw=2, label=f"h = l/{L_ELL/h:.0f} ({h*1000:.2f} um)" + ("" if broke else ", no drop yet"))
a.set_xlabel("strain"); a.set_ylabel("stress (GPa)"); a.grid(alpha=.3); a.legend(fontsize=8); a.set_title("Stress-strain for each mesh", fontsize=11)
hl = np.array([r[1] / L_ELL for r in runs]); pk = np.array([r[4] for r in runs])
# AT2 on a mesh of size h behaves as if g_c were g_c (1 + h / 2l), and LEFM peak load scales with sqrt(g_c):
#   peak(h) = peak0 * sqrt(1 + h / (2 l))   ->  fit peak0, the mesh-independent value
w = np.sqrt(1 + hl / 2); peak0 = float(np.sum(pk * w) / np.sum(w * w))
hh = np.linspace(0, max(hl) * 1.1, 100)
b.plot(hh, peak0 * np.sqrt(1 + hh / 2), "--", color="#555", label=f"AT2 mesh effect, fitted: peak0 = {peak0:.3f} GPa")
b.plot(hl, pk, "s", color="#1f5f8b", ms=8, label="our runs")
b.axhline(ppeak, color="#d9822b", lw=1.5, label=f"paper peak {ppeak:.3f} GPa")
b.set_xlabel("h / l (element size over crack width)"); b.set_ylabel("peak stress (GPa)"); b.set_xlim(0, max(hl) * 1.1)
b.grid(alpha=.3); b.legend(fontsize=8); b.set_title(f"Refining the mesh: mesh-free peak {peak0:.3f} vs paper {ppeak:.3f} GPa ({100*(peak0-ppeak)/ppeak:+.1f}%)", fontsize=10)
fig.tight_layout(); fig.savefig("fig5_mesh_study.png", dpi=200)
with open("mesh_study.txt", "w") as f:
    f.write(f"{'run':10s}{'h (um)':>10s}{'h/l':>8s}{'peak GPa':>10s}{'strain':>10s}{'vs paper':>10s}{'cracked':>9s}\n")
    for d, h, e, s, p, ep, broke in runs:
        f.write(f"{d:10s}{h*1000:10.3f}{h/L_ELL:8.3f}{p:10.4f}{ep:10.5f}{100*(p-ppeak)/ppeak:+9.1f}%{str(broke):>9s}\n")
    f.write(f"\nfitted mesh-independent peak {peak0:.4f} GPa vs paper {ppeak:.4f} GPa ({100*(peak0-ppeak)/ppeak:+.1f}%)\n")
print(open("mesh_study.txt").read())
