# Creates the gif of the moose results with the damage spreading throughout the geometry, updated per time step

"""

Works for both validation runs:
  RVE :  python3 exo_anim.py rve_h8_out.e  --pores --curve rve_h8_out.csv --paper rve --warp 20
  SENT:  python3 exo_anim.py sent_paper_out.e --mirror --curve sent_paper_out.csv --paper sent --zoom 0.45 1 0.45 0.55

Options
  --var c            nodal variable to color (default damage c)
  --pores            blank out elements whose stiffness map efac_aux < 0.01 (the explicit pores), drawn white
  --mirror           SENT half model: mirror about y = 0 and shift up by the half height, to show the full plate
  --warp K           draw the deformed shape with displacements magnified K times (0 = undeformed)
  --zoom x0 x1 y0 y1 show only this window (mesh units)
  --mesh             draw element edges (useful with --zoom to see the crack cross the mesh lattice)
  --curve FILE       the run's *_out.csv, to add a stress-strain panel with a moving marker
  --paper sent|rve   add the paper's digitized curve to that panel (read from viz_sent.py / viz_rve.py)
  --dark             dark background (pores stay white, like the paper)
  --label TEXT       legend name for our curve (default "our MOOSE run")
  --every N          use every Nth saved step (default 1)
  --fps F            frames per second (default 10)
  --out NAME         output file (default <exodus>_anim.gif; .mp4 works if ffmpeg exists)
Needs: pip install --user netCDF4 matplotlib pillow
"""
import argparse, ast, csv, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
from netCDF4 import Dataset

p = argparse.ArgumentParser()
p.add_argument("exodus"); p.add_argument("--var", default="c"); p.add_argument("--pores", action="store_true")
p.add_argument("--mirror", action="store_true"); p.add_argument("--warp", type=float, default=0.0)
p.add_argument("--zoom", type=float, nargs=4); p.add_argument("--mesh", action="store_true")
p.add_argument("--curve"); p.add_argument("--paper", choices=["sent", "rve"])
p.add_argument("--dark", action="store_true"); p.add_argument("--label", default="our MOOSE run")
p.add_argument("--every", type=int, default=1); p.add_argument("--fps", type=float, default=10); p.add_argument("--out")
a = p.parse_args()

ds = Dataset(a.exodus)
def names(var):
    if var not in ds.variables: return []
    return ["".join(ch.decode() for ch in row if ch not in (b"", b"\x00")).strip() for row in ds.variables[var][:].data]
nod = names("name_nod_var"); elem = names("name_elem_var")
def nodal(name):
    k = nod.index(name) + 1
    return ds.variables[f"vals_nod_var{k}"][:] if f"vals_nod_var{k}" in ds.variables else ds.variables["vals_nod_var"][:, k - 1, :]
x, y = ds.variables["coordx"][:].data, ds.variables["coordy"][:].data
conn = np.vstack([ds.variables[v][:].data for v in ds.variables if re.fullmatch(r"connect\d+", v)]) - 1
times = ds.variables["time_whole"][:].data
C = np.asarray(nodal(a.var))
ux = np.asarray(nodal("disp_x")) if a.warp and "disp_x" in nod else None
uy = np.asarray(nodal("disp_y")) if a.warp and "disp_y" in nod else None
pore_elem = np.zeros(len(conn), bool)
if a.pores and "efac_aux" in elem:
    k = elem.index("efac_aux") + 1
    v = [vv for vv in ds.variables if re.fullmatch(rf"vals_elem_var{k}eb\d+", vv)]
    pore_elem = np.concatenate([np.asarray(ds.variables[vv][0]) for vv in v]) < 0.01
# quads -> triangles; mask triangles that belong to pore elements
if conn.shape[1] == 4:
    tris = np.vstack([conn[:, [0, 1, 2]], conn[:, [0, 2, 3]]]); tmask = np.concatenate([pore_elem, pore_elem])
else:
    tris = conn[:, :3]; tmask = pore_elem
H = y.max()
def coords(i):
    X, Y = x.copy(), y.copy()
    if ux is not None: X = X + a.warp * ux[i]; Y = Y + a.warp * uy[i]
    return X, Y
def frames_xy(i):
    X, Y = coords(i)
    if not a.mirror: return [(X, Y)]
    return [(X, Y + H), (X, H - Y)]   # top half and its mirror image

curve = None
if a.curve and os.path.exists(a.curve):
    with open(a.curve) as f:
        r = csv.reader(f); h = next(r); A = np.array([[float(v) for v in row] for row in r if row])
    curve = (A[:, h.index("time")], A[:, h.index("strain")], A[:, h.index("stress")])
paper = None
if a.paper:
    src = open("viz_sent.py" if a.paper == "sent" else "viz_rve.py").read()
    m = re.search(r"^(PAPER4?) = (\[.*\])$", src, re.M); paper = np.array(ast.literal_eval(m.group(2))); paper = paper[paper[:, 1] > 0]

steps = list(range(0, len(times), max(1, a.every)))
if a.dark: plt.style.use("dark_background")
PORE = "white"
if curve is not None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw=dict(width_ratios=[1.15, 1]))
else:
    fig, ax = plt.subplots(figsize=(6.5, 6)); bx = None
if bx is not None:
    if paper is not None: bx.plot(paper[:, 0], paper[:, 1], "o", ms=2.5, color="#d9822b", label="Chakraborty et al. 2016")
    bx.plot(curve[1], curve[2], color="#5fb3e8" if a.dark else "#1f5f8b", lw=2, label=a.label)
    dot, = bx.plot([], [], "o", color="w" if a.dark else "k", ms=7)
    bx.set_xlabel("strain"); bx.set_ylabel("stress (GPa)" if a.paper == "sent" else "stress (MPa)"); bx.grid(alpha=.25); bx.legend(fontsize=9)

# fixed view: the extent over the first and last frames, so the picture does not jump around
_xy = frames_xy(0) + frames_xy(len(times) - 1)
LIMS = ((min(X.min() for X, _ in _xy), max(X.max() for X, _ in _xy)), (min(Y.min() for _, Y in _xy), max(Y.max() for _, Y in _xy)))
def draw(i):
    ax.clear()
    for X, Y in frames_xy(i):
        tri = mtri.Triangulation(X, Y, tris, mask=tmask)
        im = ax.tripcolor(tri, C[i], shading="gouraud", cmap="jet", vmin=0, vmax=1)
        if a.mesh:
            q = conn if conn.shape[1] == 4 else conn[:, :3]
            ax.add_collection(PolyCollection(np.stack([X[q], Y[q]], -1), facecolor="none", edgecolor="k", linewidth=0.15, alpha=0.5))
    ax.set_aspect("equal"); ax.set_facecolor(PORE)          # masked (pore) triangles show this color
    if a.zoom: ax.set_xlim(a.zoom[0], a.zoom[1]); ax.set_ylim(a.zoom[2], a.zoom[3])
    else:
        ax.set_xlim(*LIMS[0]); ax.set_ylim(*LIMS[1])
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    lab = f"damage {a.var}" + (f", strain {np.interp(times[i], curve[0], curve[1]):.4f}" if curve is not None else f", t = {times[i]:.4g}") + (f", deformation x{a.warp:g}" if a.warp else "")
    ax.set_title(lab, fontsize=10)
    if bx is not None:
        e = np.interp(times[i], curve[0], curve[1]); s = np.interp(times[i], curve[0], curve[2]); dot.set_data([e], [s])
    return im
im = draw(steps[0]); fig.colorbar(im, ax=ax, shrink=.75, label="damage c")
fig.tight_layout(pad=1.2, rect=(0, 0, 1, 0.95))
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
out = a.out or os.path.splitext(a.exodus)[0] + "_anim.gif"
writer = FFMpegWriter(fps=a.fps) if out.endswith(".mp4") else PillowWriter(fps=a.fps)
FuncAnimation(fig, lambda k: draw(steps[k]), frames=len(steps)).save(out, writer=writer, dpi=110, savefig_kwargs=dict(facecolor=fig.get_facecolor()))
# also save the last frame as a still
draw(steps[-1]); fig.savefig(os.path.splitext(out)[0] + "_last.png", dpi=200, facecolor=fig.get_facecolor())
print(f"wrote {out} ({len(steps)} frames) and {os.path.splitext(out)[0]}_last.png")
