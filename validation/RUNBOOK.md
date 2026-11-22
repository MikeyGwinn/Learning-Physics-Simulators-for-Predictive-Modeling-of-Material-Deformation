# Runbook: one dataset run from your own HPC account

One run = 1 × 1 mm graphite RVE, 8 µm mesh, pulled to 0.4% strain. About 1 h on 12 cores. Output: 60 `.npz` frames (see `export/FORMAT.md`).

## One-time setup

- `module load slurm` (add it to `~/.bashrc`)
- `pip install --user numpy netCDF4 matplotlib`
- Clone the repo into `/nfs/hpc/share/$USER/lps`
- Check out Gunnar's pore generator (PR #24 branch) into `/nfs/hpc/share/$USER/lps-pores`
- Check you can read the shared files: `ls /nfs/hpc/share/brockert/capstone/moose/moose.sif` (if "Permission denied", tell Tiago)

## One run

- Pick your seed from the manifest (dataset seeds start at 101) and make a folder:
  ```bash
  S=101; C=/nfs/hpc/share/brockert/capstone
  W=/nfs/hpc/share/$USER/lps_dataset/runs/s1-0$S; mkdir -p $W && cd $W
  cp $C/validation/rve_paper/rve_paper.i $C/validation/rve_paper/gunnar_to_moose.py .
  ```
- Make the pores (if it says "could not place pore", skip this seed and use the next one):
  ```bash
  (cd /nfs/hpc/share/$USER/lps-pores && python3 -m pore_generator generate --seed $S --clearance-um 4 --output $W/samples/seed-$S)
  python3 gunnar_to_moose.py samples/seed-$S
  ```
- Make the input and submit it:
  ```bash
  sed "s/^seed  = 1/seed  = $S/; s/^h     = 4.0/h     = 8.0/" rve_paper.i > rve_h8_s$S.i
  sbatch -p share,dgx2 -c 12 --mem=16G -t 3:00:00 /nfs/hpc/share/$USER/lps/scripts/moose_container.sbatch $PWD/rve_h8_s$S.i
  ```

## Check on it

- Running? `squeue -u $USER`
- Errors (if it left the queue early): `grep -a -A6 '\*\*\* ERROR' moose_*.log`
- Done when the log ends with `Finished Executing`

## Export

```bash
E=/nfs/hpc/share/brockert/capstone/validation/export
python3 $E/export_npz.py rve_h8_s${S}_out.e --unit um --out export --points 4096 --frames 60
python3 $E/check_npz.py export
```

- `check_npz.py` must print OK.

## Rules

- Write only in your own folder: `/nfs/hpc/share/$USER/lps_dataset/`
- Never rerun a failed seed with different settings
- No Dr. Chen data in the public repo or in cloud AI tools
