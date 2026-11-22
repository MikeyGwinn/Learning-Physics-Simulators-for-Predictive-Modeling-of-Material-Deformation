"""Proposed normalization for lps-npz-v1 runs (numpy only, mirrors what the torch loader should do).

  python3 normalize_ref.py sample_v1/*        # prints the stats below for the given runs

One training sample = one frame pair (t -> t+1) of one run.
  input   pos[t]                  (N,3) mm, already 0..1, used as is
          eps[t], deps            applied strain and strain step, both divided by EPS_SCALE
  target  (pos[t+1]-pos[t])/deps  (N,3) mm per unit strain: ~0.5-0.7 RMS before the crack, up to ~10x while it runs
  rollout pos[t+1] = pos[t] + deps * model_output
Split train/test by run (pore seed), never by frame.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_npz import load_run

EPS_SCALE = 0.004          # largest applied strain in the dataset -> eps input in 0..1

def pairs(run):
    """Yield (x_pos, cond, target, is_post_peak) for every frame pair of one run."""
    pos, eps, pk = run["pos"], run["strain"], run["meta"]["peak_frame"]
    for t in range(len(eps) - 1):
        deps = eps[t + 1] - eps[t]
        cond = np.array([eps[t] / EPS_SCALE, deps / EPS_SCALE], np.float32)
        yield pos[t], cond, (pos[t + 1] - pos[t]) / deps, t >= pk

def denormalize(pos_t, deps, model_output):
    return pos_t + deps * model_output

if __name__ == "__main__":
    pre, post = [], []
    for d in sys.argv[1:]:
        for _, _, y, after in pairs(load_run(d)):
            (post if after else pre).append(y[:, :2])
    pre, post = np.concatenate(pre), np.concatenate(post)
    print("target std  x, y   before peak:", pre.std(0).round(3), "  after peak:", post.std(0).round(3))
    print("target max |.| x, y before peak:", np.abs(pre).max(0).round(2), "  after peak:", np.abs(post).max(0).round(2))
