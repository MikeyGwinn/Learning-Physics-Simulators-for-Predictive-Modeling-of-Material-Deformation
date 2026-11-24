# MOOSE validation: replicating Chakraborty, Sabharwall & Carroll (2016)

Phase-field fracture runs in MOOSE that reproduce the benchmarks in
P. Chakraborty, P. Sabharwall, M. C. Carroll, "A phase-field approach to model multi-axial and
microstructure dependent fracture in nuclear grade graphite," J. Nucl. Mater. 475 (2016).

Everything runs on the OSU HPC through INL's MOOSE container (`scripts/moose_container.sbatch`).

## What is here

| Path | What it does |
|---|---|
| `sent/sent_paper.i` | SENT benchmark (paper Sec. 2, Fig. 1, Table 1): 1 x 1 mm notched plate, half model by symmetry. Parameters `h` (element size at the crack plane) and `bias` (mesh grading) at the top of the file |
| `sent/viz_sent.py` | Stress-strain vs the paper's Fig. 1b, damage profile across the crack vs Eq. 1, damage maps, crack-tip position, GIF |
| `sent/compare_sent.py` | Mesh study (Check 2): peak stress vs h/l for several runs, extrapolated to h -> 0 |
| `rve/rve_paper.i` | H-451 graphite RVE (paper Sec. 3-4.1, Figs. 3-5, Table 2): 1 x 1 mm, porosity 0.20, pores as very low stiffness regions |
| `rve/gunnar_to_moose.py` | Turns a sample from the team's pore generator (`pore_generator/`, PR #24) into the stiffness map the RVE input reads. Does not modify the generator's outputs |
| `rve/viz_rve.py` | Stress-strain vs the paper's Fig. 4, damage maps with pores (Fig. 5), GIF |
| `exo_anim.py` | Animation of any run from its Exodus file, every saved step; options for pores, mirrored half models, magnified deformation, zoom with mesh edges, dark mode |
| `paper_digitized/` | Curves digitized from the accepted manuscript (Fig. 1b, Fig. 4), used as comparison targets |

## How to run (RVE example)

```bash
# 1. microstructure from the pore generator (PR #24 branch)
python3 -m pore_generator generate --seed 1 --clearance-um 4 --output samples/seed-1
# 2. stiffness map for MOOSE
python3 gunnar_to_moose.py samples/seed-1            # writes efactor_1.txt
# 3. check the input without the queue, then submit
combined-opt -i rve_paper.i --check-input           # inside the container
sbatch -p share,dgx2 -c 16 -t 12:00:00 scripts/moose_container.sbatch $PWD/rve_paper.i
# 4. figures and animation
python3 viz_rve.py rve_paper 1
python3 ../exo_anim.py rve_paper_out.e --pores --curve rve_paper_out.csv --paper rve --warp 20 --dark
```

## Model: paper vs this input

| Setting | Paper | Here | Status |
|---|---|---|---|
| Energy | AT2, spectral split (Eqs. 5, 8) | `strain_spectral`, `c^2 g_c / 2l` | match |
| Damage law | eta dc/dt = <l lap c + 2(1-c) psi+/g_c - c/l>+ (Eq. 9) | Same terms; `visco` = eta | match |
| Irreversibility | Macaulay bracket on dc/dt | History variable (max psi+) | differs |
| SENT material | E 207 GPa, nu 0.3, g_c 0.0027 GPa mm, l 0.0075 mm, k 1e-6 | Same | match |
| RVE material | E 7.6 GPa, nu 0.18, g_c 10.68 MPa um, l 10 um, eta 1e-3, h 4 um | Same | match |
| Pores | log-normal radii (mu 2.941, sigma 0.703); r >= 20 um explicit, smaller smeared in 50 um cells | Team pore generator; pore stiffness 1e-4 E | match / assumed |
| Solver | Monolithic, backward Euler, PJFNK, ASM + LU | Same | match |
| Loading rate, plane strain/stress | Not stated | 1 (length unit)/s, plane strain | assumed |

## Results so far (Nov 5, 2026)

| Check | Paper | Ours | Note |
|---|---|---|---|
| SENT initial slope | 139.3 GPa | 139.2 GPa | -0.1% |
| SENT peak | 0.70 GPa | no fracture by strain 0.0058 at h = l/2 | mesh study running |
| RVE initial slope (8 um test mesh, seed 1) | 4.26 GPa | 4.43 GPa | +4% |
| RVE peak (8 um test mesh, seed 1) | 10.7 MPa | 12.2 MPa | +14%, expected from the coarse mesh |
| RVE crack pattern | cracks link pores, roughly normal to the load (Fig. 5) | same qualitative pattern | see `results/` |

Known effect: on a mesh of size h the AT2 model behaves as if g_c were about g_c (1 + h/2l), so coarse meshes
give peaks that are too high. The 4 um and 2 um RVE runs and the three-mesh SENT study test this.

## Next

- Finish the mesh studies; compare seeds 1-3 with the paper's three RVEs (Fig. 6).
- Same RVE with pores as holes, meshed by SEMtoMesh, against pores as soft regions (RFC check 3, multi-pore).
- Real ET-10 microstructures: SEM image -> SEMtoMesh -> MOOSE -> data format v0.
