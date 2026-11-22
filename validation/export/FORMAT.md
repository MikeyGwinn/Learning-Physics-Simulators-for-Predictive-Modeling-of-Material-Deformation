# Simulation dataset format: lps-npz-v1

One folder per simulation run and one `.npz` file per frame. A frame is one load step: the sample is pulled a little further each frame, so the frame number follows the applied strain, not time.

```
sample_v1/
  rve_h8/                   one MOOSE run (pore layout seed 1, 8 um mesh)
    frame_0000.npz          unloaded sample
    frame_0001.npz          ...
    frame_0059.npz          after the crack has gone through
    points.npz              the fixed point set (same points, same order, every frame)
    meta.json               what each frame is: strain, stress, peak frame, units
    check.png               picture of the exported points (sanity check)
```

## Each frame file

| Key | Shape | Type | Meaning |
|---|---|---|---|
| `means3D` | (1, N, 3) | float32 | point positions in **mm**, z = 0 (2D runs) |
| `log_scales` | (1, N, 3) | float32 | placeholder, zeros |
| `logit_opacities` | (1, N, 1) | float32 | placeholder, zeros |
| `shs` | (1, N, 16, 3) | float32 | placeholder, zeros |
| `unnorm_rotations` | (1, N, 4) | float32 | placeholder, identity quaternion (1, 0, 0, 0) |
| `disp` | (1, N, 3) | float32 | extra: displacement from frame 0, mm (`means3D = rest + disp`) |
| `damage` | (1, N, 1) | float32 | extra: crack field, 0 = intact, 1 = fully cracked |

N = 4096 points in every frame of every run. The placeholders are stored compressed, so they take almost no space.

## points.npz and meta.json

- `points.npz`:
  - `rest` (N, 3): positions in frame 0
  - `node_id`: MOOSE node numbers
  - `is_top`, `is_bottom`, `is_left`, `is_right`: points on the sample edges
  - `stiffness`: local E/E0, near 0 next to pores
- `meta.json`:
  - `frame_strain`, `frame_stress`: the load and the sample's response for every frame
  - `peak_frame`: the frame at maximum stress, right before the crack runs
  - `units`, `domain_mm`, source file paths

## Things the loader needs to handle

1. **Displacements are tiny.** The whole sample stretches about 4 um over a 1 mm sample (0.4%). One frame to the next moves points about 0.03 um. A loss on raw positions is therefore almost zero even for a model that predicts no motion. Train on `disp` (or position minus frame 0), divided by its standard deviation over the training set.
2. **Frames are evenly spaced in applied strain.** Before the peak, every frame is the same displacement pattern scaled up (linear elasticity). The crack happens within 3-5 frames after `peak_frame`. If that is too few, the exporter can add extra frames after the peak (`--crack-frames 10`). Each frame's strain is listed in `meta.json`.
3. **Use strain, not force, as the load variable.** After the peak the force drops again, so two different frames can have the same force. Strain always increases.
4. **The crack is thinly sampled.** Only 11-37 of the 4096 points end up inside the crack (`damage` > 0.9). The crack shows mainly as a jump in `disp` between neighbouring points.

## Sample set v1 (6 runs, 30 MB)

| Run | Pore seed | Mesh | Peak stress (MPa) | Peak frame |
|---|---|---|---|---|
| rve_h8 | 1 | 8 um | 12.18 | 46 |
| rve_h8_s2 | 2 | 8 um | 9.97 | 47 |
| rve_h8_s3 | 3 | 8 um | 9.52 | 40 |
| rve_paper | 1 | 4 um | 11.78 | 51 |
| rve_s2 | 2 | 4 um | 9.80 | 51 |
| rve_s3 | 3 | 4 um | 9.62 | 52 |

The planned dataset uses the 8 um mesh (about 1 h per run, within 1-3% of the 4 um result).

## Reading it

```python
from check_npz import load_run           # reference reader shipped with the data
r = load_run("sample_v1/rve_h8")
r["pos"]     # (60, 4096, 3) positions per frame, mm
r["disp"]    # (60, 4096, 3) displacement per frame, mm
r["damage"]  # (60, 4096)    crack field per frame
r["strain"]  # (60,)         applied strain per frame (the load)
```

`python3 check_npz.py sample_v1/*` checks every run (shapes, point order, z = 0, placeholders) and redraws `check.png`.
