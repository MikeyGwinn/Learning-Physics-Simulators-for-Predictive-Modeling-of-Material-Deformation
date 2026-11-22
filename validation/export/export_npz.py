"""Export one MOOSE run (Exodus) into the model's per-frame .npz format (lps-npz-v1).

  python3 export_npz.py RUN_out.e --csv RUN_out.csv --unit um --out dataset/RUN
  python3 export_npz.py rve_h8_out.e --csv rve_h8_out.csv --unit um --out dataset/rve_h8_s1 --points 4096 --frames 60

Writes into --out:
  frame_0000.npz ...   one file per load step. Keys the model reads:
                         means3D (1,N,3) float32  point positions in mm, z = 0
                         log_scales (1,N,3), logit_opacities (1,N,1), shs (1,N,16,3), unnorm_rotations (1,N,4)
                         placeholders (zeros; rotation = identity quaternion), stored compressed so they cost ~nothing
                       Extra keys (ignore if unused):
                         disp (1,N,3) float32     displacement from the rest shape, mm (no round-off from subtracting positions)
                         damage (1,N,1) float32   phase-field damage c, 0 intact .. 1 cracked
  points.npz           the fixed point set, same order in every frame: rest (N,3) mm, node_id (N,),
                       is_top / is_bottom / is_left / is_right (N,) bool, stiffness (N,) E/E0 near the node
  meta.json            units, frame -> applied strain / stress / time, peak frame, source files, format version

Frames are evenly spaced in applied strain (the load), not in solver time; each frame is the saved
MOOSE step nearest to its target strain. Points are a fixed random subset of the mesh nodes, the same
in every frame; nodes that sit only in pores (stiffness map < 0.01) are left out.
Needs: pip install --user netCDF4 numpy
"""
import argparse, csv, json, os, re
import numpy as np
from netCDF4 import Dataset

FORMAT = "lps-npz-v1"
p = argparse.ArgumentParser()
p.add_argument("exodus")
p.add_argument("--csv", help="the run's *_out.csv (strain, stress per step); default: exodus name with .csv")
p.add_argument("--unit", choices=["um", "mm"], required=True, help="length unit of the MOOSE run (RVE: um, SENT: mm)")
p.add_argument("--out", required=True)
p.add_argument("--points", type=int, default=4096, help="points per frame (0 = all solid nodes)")
p.add_argument("--frames", type=int, default=60, help="frames from zero load to the end of the run")
p.add_argument("--stop-after-drop", type=float, default=0.2,
               help="end the frames once stress falls below this fraction of the peak (0 = use the whole run)")
p.add_argument("--crack-frames", type=int, default=0,
               help="extra frames spread between the peak and the end (the crack happens in a few saved steps; "
                    "uniform spacing alone gives it only ~3-5 frames). meta.json records each frame's strain")
p.add_argument("--seed", type=int, default=0, help="seed for the point subset")
p.add_argument("--no-placeholders", action="store_true", help="write only means3D/disp/damage")
a = p.parse_args()
TO_MM = 1e-3 if a.unit == "um" else 1.0

ds = Dataset(a.exodus)
def names(var):
    if var not in ds.variables: return []
    return ["".join(ch.decode() for ch in row if ch not in (b"", b"\x00")).strip() for row in ds.variables[var][:].data]
nod, elem = names("name_nod_var"), names("name_elem_var")
def nodal(name):
    k = nod.index(name) + 1
    v = f"vals_nod_var{k}"
    return np.asarray(ds.variables[v][:] if v in ds.variables else ds.variables["vals_nod_var"][:, k - 1, :])
X0 = np.stack([np.asarray(ds.variables["coordx"][:]), np.asarray(ds.variables["coordy"][:])], 1)
blocks = sorted([v for v in ds.variables if re.fullmatch(r"connect\d+", v)], key=lambda v: int(v[7:]))
conn = np.vstack([np.asarray(ds.variables[v][:]) for v in blocks]) - 1
times = np.asarray(ds.variables["time_whole"][:])
UX, UY = nodal("disp_x"), nodal("disp_y")
C = nodal("c") if "c" in nod else np.zeros_like(UX)
nn = len(X0)

# stiffness near each node: the largest E/E0 among the elements around it (1 everywhere for a conformal mesh)
stiff = np.ones(nn)
if "efac_aux" in elem:
    k = elem.index("efac_aux") + 1
    ev = np.concatenate([np.asarray(ds.variables[f"vals_elem_var{k}eb{b[7:]}"][0]) for b in blocks])
    stiff = np.zeros(nn)
    np.maximum.at(stiff, conn.ravel(), np.repeat(ev, conn.shape[1]))
solid = np.flatnonzero(stiff >= 0.01)

# fixed point subset, sorted by node id so the order is reproducible
rng = np.random.default_rng(a.seed)
ids = solid if a.points <= 0 or a.points >= len(solid) else np.sort(rng.choice(solid, a.points, replace=False))
N = len(ids)

# load history from the CSV (time = applied top displacement in these inputs)
csvp = a.csv or os.path.splitext(a.exodus)[0] + ".csv"
with open(csvp) as f:
    r = csv.reader(f); head = next(r); A = np.array([[float(v) for v in row] for row in r if row])
ct, ce, cs = A[:, head.index("time")], A[:, head.index("strain")], A[:, head.index("stress")]
e_of_t = lambda t: np.interp(t, ct, ce); s_of_t = lambda t: np.interp(t, ct, cs)
ipk = int(np.argmax(cs)); t_end = ct[-1]
if a.stop_after_drop > 0:
    after = np.flatnonzero((np.arange(len(cs)) > ipk) & (cs < a.stop_after_drop * cs[ipk]))
    if len(after): t_end = ct[after[0]]
t_end = min(t_end, times[-1])

# frames evenly spaced in strain; take the saved step nearest each target, no repeats
targets = np.linspace(e_of_t(0.0), e_of_t(t_end), a.frames)
if a.crack_frames > 0:
    targets = np.unique(np.concatenate([targets, np.linspace(ce[ipk], e_of_t(t_end), a.crack_frames + 2)[1:-1]]))
e_steps = e_of_t(times)
steps = []
for e in targets:
    i = int(np.argmin(np.abs(e_steps - e)))
    if times[i] <= t_end + 1e-12 and (not steps or i != steps[-1]): steps.append(i)

os.makedirs(a.out, exist_ok=True)
rest = np.zeros((N, 3), np.float32); rest[:, :2] = X0[ids] * TO_MM
z = lambda *s: np.zeros(s, np.float32)
for f_i, st in enumerate(steps):
    u = np.zeros((1, N, 3), np.float32); u[0, :, 0] = UX[st, ids] * TO_MM; u[0, :, 1] = UY[st, ids] * TO_MM
    out = dict(means3D=(rest[None] + u).astype(np.float32), disp=u, damage=C[st, ids].astype(np.float32)[None, :, None])
    if not a.no_placeholders:
        q = z(1, N, 4); q[..., 0] = 1.0
        out.update(log_scales=z(1, N, 3), logit_opacities=z(1, N, 1), shs=z(1, N, 16, 3), unnorm_rotations=q)
    np.savez_compressed(os.path.join(a.out, f"frame_{f_i:04d}.npz"), **out)

x, y = X0[:, 0], X0[:, 1]; tol = 1e-9 * max(np.ptp(x), np.ptp(y))
np.savez_compressed(os.path.join(a.out, "points.npz"), rest=rest, node_id=ids,
                    is_top=np.abs(y[ids] - y.max()) < tol, is_bottom=np.abs(y[ids] - y.min()) < tol,
                    is_left=np.abs(x[ids] - x.min()) < tol, is_right=np.abs(x[ids] - x.max()) < tol,
                    stiffness=stiff[ids].astype(np.float32))
tt = times[steps]; ss = s_of_t(tt)
meta = dict(format=FORMAT, units=dict(length="mm", stress="MPa" if a.unit == "um" else "GPa"),
            source=dict(exodus=os.path.abspath(a.exodus), csv=os.path.abspath(csvp)),
            n_points=int(N), n_frames=len(steps), n_mesh_nodes=int(nn), n_solid_nodes=int(len(solid)), point_seed=a.seed,
            frame_index_variable="applied strain (top displacement / height)",
            frame_strain=[float(v) for v in e_of_t(tt)], frame_stress=[float(v) for v in ss], frame_time=[float(v) for v in tt],
            moose_step=[int(v) for v in steps], peak_stress=float(cs[ipk]), peak_strain=float(ce[ipk]),
            peak_frame=int(np.argmax(ss)), domain_mm=[float(np.ptp(x) * TO_MM), float(np.ptp(y) * TO_MM)])
with open(os.path.join(a.out, "meta.json"), "w") as f: json.dump(meta, f, indent=1)
print(f"{a.out}: {len(steps)} frames x {N} points (of {len(solid)} solid nodes), strain 0 -> {meta['frame_strain'][-1]:.5f}, "
      f"peak {meta['peak_stress']:.4g} at frame {meta['peak_frame']}")
