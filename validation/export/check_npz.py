"""Reference reader + check for lps-npz-v1 runs. The model team's loader only has to do what load_run() does.

  python3 check_npz.py dataset/rve_h8_s1            # checks shapes and ordering, writes dataset/rve_h8_s1/check.png
  python3 check_npz.py dataset/*                     # several runs

In Python:  from check_npz import load_run;  r = load_run("dataset/rve_h8_s1")
  r["pos"] (F,N,3)  r["disp"] (F,N,3)  r["damage"] (F,N)  r["strain"] (F,)  r["rest"] (N,3)  r["meta"] dict
"""
import glob, json, os, sys
import numpy as np

def load_run(d):
    meta = json.load(open(os.path.join(d, "meta.json")))
    files = sorted(glob.glob(os.path.join(d, "frame_*.npz")))
    fr = [np.load(f) for f in files]
    pts = np.load(os.path.join(d, "points.npz"))
    return dict(meta=meta, rest=pts["rest"], is_top=pts["is_top"], stiffness=pts["stiffness"],
                pos=np.concatenate([f["means3D"] for f in fr]), disp=np.concatenate([f["disp"] for f in fr]),
                damage=np.concatenate([f["damage"][..., 0] for f in fr]), strain=np.array(meta["frame_strain"]),
                stress=np.array(meta["frame_stress"]), keys=sorted(fr[0].files))

def check(d):
    r = load_run(d); F, N, _ = r["pos"].shape; m = r["meta"]; ok = True
    def need(cond, msg):
        nonlocal ok
        if not cond: ok = False; print("  FAIL:", msg)
    need(m["format"] == "lps-npz-v1", "format tag")
    need(F == m["n_frames"] and N == m["n_points"], "frame/point counts match meta.json")
    need(r["pos"].dtype == np.float32, "means3D is float32")
    need(np.allclose(r["pos"][:, :, 2], 0), "z = 0")
    need(np.allclose(r["pos"], r["rest"][None] + r["disp"], atol=1e-6), "means3D = rest + disp (same point order)")
    need(np.all(np.diff(r["strain"]) > 0), "strain increases every frame")
    need(np.abs(r["disp"][0]).max() < 1e-9, "frame 0 is the unloaded shape")
    for k in ["log_scales", "logit_opacities", "shs", "unnorm_rotations"]:
        need(k in r["keys"], f"placeholder {k} present")
    u = np.linalg.norm(r["disp"], axis=-1)
    print(f"{d}: {F} frames x {N} points, domain {m['domain_mm']} mm, strain to {r['strain'][-1]:.5f}, "
          f"max |u| {u.max()*1e3:.3g} um ({u.max()/max(m['domain_mm']):.1e} of the sample), "
          f"median frame-to-frame |du| {np.median(np.linalg.norm(np.diff(r['disp'],axis=0),axis=-1))*1e3:.3g} um, "
          f"cracked points (c>0.9) at end {int((r['damage'][-1]>0.9).sum())}  -> {'OK' if ok else 'PROBLEMS'}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        k = m["peak_frame"]; picks = [0, k, F - 1]
        warp = 0.05 * max(m["domain_mm"]) / max(u.max(), 1e-30)   # magnify so the largest motion is 5% of the sample
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
        for a_, i in zip(ax[:3], picks):
            P = r["rest"] + warp * r["disp"][i]
            sc = a_.scatter(P[:, 0], P[:, 1], c=r["damage"][i], s=1.5, cmap="jet", vmin=0, vmax=1)
            a_.set_aspect("equal"); a_.set_xticks([]); a_.set_yticks([])
            a_.set_title(f"frame {i}: strain {r['strain'][i]:.4f}" + (" (peak)" if i == k else ""), fontsize=10)
        fig.colorbar(sc, ax=ax[2], shrink=.8, label="damage c")
        ax[3].plot(r["strain"], r["stress"], "o-", ms=3); ax[3].plot(r["strain"][k], r["stress"][k], "rs")
        ax[3].set_xlabel("applied strain (frame index variable)"); ax[3].set_ylabel(f"stress ({m['units']['stress']})")
        ax[3].set_title("one dot = one exported frame", fontsize=10); ax[3].grid(alpha=.3)
        fig.suptitle(f"{os.path.basename(d.rstrip('/'))}: exported points, displacement x{warp:.0f}", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(d, "check.png"), dpi=130); plt.close(fig)
    except ImportError:
        pass
    return ok

if __name__ == "__main__":
    sys.exit(0 if all([check(d) for d in sys.argv[1:]]) else 1)
